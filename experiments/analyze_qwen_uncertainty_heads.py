#!/usr/bin/env python3
"""Locate Qwen heads that linearly encode sample-level LGD U_A/U_D/U_E.

The script ranks every ``(layer, head)`` with image-grouped out-of-fold Ridge
probes, selects heads using only the train split, and reports fully held-out
validation performance.  It compares the answer-free prompt-final view against
the target model's existing greedy-answer last-token view.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Mapping, Sequence

from joblib import Parallel, delayed
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch


TARGETS = ("u_a", "u_d", "u_e")
TOTAL_TARGET = "u_total"
AVAILABLE_TARGETS = (*TARGETS, TOTAL_TARGET)
DEFAULT_VIEWS = ("answer_free", "greedy_answer")
NEGATIVE_TARGET_TOLERANCE = 1e-12
SCRIPT_DIR = Path(__file__).resolve().parent


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False
        )
    os.replace(temporary, path)


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_manifest(features_dir: Path, split: str) -> Dict[str, Any]:
    path = Path(features_dir) / split / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run collect_qwen_uncertainty_heads.py for {split}."
        )
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest {path} contains no records.")
    return manifest


def _metadata_arrays(
    manifest: Mapping[str, Any], targets: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = manifest["records"]
    sample_ids = np.asarray([str(row["sample_id"]) for row in records])
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs in the feature manifest.")
    groups = np.asarray([str(row.get("group_id", row["sample_id"])) for row in records])
    values = np.asarray([
        [
            (
                sum(float(row["labels"][component]) for component in TARGETS)
                if target == TOTAL_TARGET
                else float(row["labels"][target])
            )
            for target in targets
        ]
        for row in records
    ], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Uncertainty targets must all be finite.")
    if bool((values < -NEGATIVE_TARGET_TOLERANCE).any()):
        raise ValueError("LGD uncertainty targets must be non-negative.")
    # KL-based targets are non-negative in theory, but roundoff can produce
    # values just below zero (for example, -1e-16 for identical distributions).
    values = np.maximum(values, 0.0)
    return sample_ids, groups, values


def load_view_features(
    features_dir: Path,
    manifest: Mapping[str, Any],
    *,
    split: str,
    view: str,
) -> np.ndarray:
    tensors = []
    expected = tuple(int(value) for value in manifest["feature_shape"])
    for row in manifest["records"]:
        relative = row.get("feature_files", {}).get(view)
        if not relative:
            raise ValueError(
                f"Sample {row['sample_id']!r} has no {view!r} feature."
            )
        path = Path(features_dir) / split / relative
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if str(saved.get("sample_id")) != str(row["sample_id"]):
            raise ValueError(f"Feature/sample ID mismatch at {path}.")
        tensor = torch.as_tensor(saved["feature"]).float()
        if tuple(tensor.shape) != expected or not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"Invalid feature at {path}: expected {expected}, got "
                f"{tuple(tensor.shape)}."
            )
        tensors.append(tensor.numpy())
    return np.stack(tensors).astype(np.float32, copy=False)


def _transform_targets(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if transform == "log1p":
        return np.log1p(values)
    if transform == "none":
        return values.copy()
    raise ValueError(f"Unsupported target transform: {transform}")


def _inverse_targets(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if transform == "log1p":
        return np.maximum(np.expm1(values), 0.0)
    if transform == "none":
        return values
    raise ValueError(f"Unsupported target transform: {transform}")


def _safe_correlation(left: np.ndarray, right: np.ndarray, kind: str) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.unique(left[valid]).size < 2 or np.unique(right[valid]).size < 2:
        return float("nan")
    function = spearmanr if kind == "spearman" else pearsonr
    return float(function(left[valid], right[valid]).statistic)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    return (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2 else float("nan")
    )


def _metric_row(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    high_threshold: float,
) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    return {
        "spearman": _safe_correlation(predictions, targets, "spearman"),
        "pearson": _safe_correlation(predictions, targets, "pearson"),
        "r2": float(r2_score(targets, predictions)),
        "mae": float(mean_absolute_error(targets, predictions)),
        "high_auroc": _safe_auc(targets >= float(high_threshold), predictions),
    }


def grouped_folds(groups: Sequence[Any], num_folds: int):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    folds = min(int(num_folds), int(unique.size))
    if folds < 2:
        raise ValueError("At least two independent image groups are required.")
    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(groups), dtype=np.float32)
    return list(splitter.split(dummy, groups=groups))


def _one_head_oof(
    features: np.ndarray,
    transformed_targets: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    layer: int,
    head: int,
    ridge_alpha: float,
) -> tuple[int, int, np.ndarray]:
    values = np.asarray(features[:, layer, head], dtype=np.float64)
    predictions = np.full_like(transformed_targets, np.nan, dtype=np.float64)
    for train_index, validation_index in folds:
        scaler = StandardScaler().fit(values[train_index])
        model = Ridge(alpha=float(ridge_alpha)).fit(
            scaler.transform(values[train_index]), transformed_targets[train_index]
        )
        predictions[validation_index] = model.predict(
            scaler.transform(values[validation_index])
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"OOF predictions are incomplete for L{layer}H{head}.")
    return layer, head, predictions


def fit_oof_head_probes(
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[Any],
    *,
    target_names: Sequence[str] = TARGETS,
    num_folds: int = 5,
    ridge_alpha: float = 10.0,
    high_quantile: float = 0.75,
    target_transform: str = "log1p",
    jobs: int = 1,
) -> tuple[Dict[str, list[Dict[str, Any]]], Dict[str, float], list]:
    """Return per-target rankings from grouped multi-output Ridge probes."""
    values = np.asarray(features, dtype=np.float32)
    target_values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("features must be [samples, layers, heads, head_dim].")
    if target_values.shape != (values.shape[0], len(target_names)):
        raise ValueError("targets must be [samples, len(target_names)].")
    folds = grouped_folds(groups, num_folds)
    transformed = _transform_targets(target_values, target_transform)
    layers, heads = values.shape[1:3]
    results = Parallel(n_jobs=int(jobs), prefer="threads")(
        delayed(_one_head_oof)(
            values,
            transformed,
            folds,
            layer=layer,
            head=head,
            ridge_alpha=ridge_alpha,
        )
        for layer in range(layers)
        for head in range(heads)
    )
    thresholds = {
        name: float(np.quantile(target_values[:, index], float(high_quantile)))
        for index, name in enumerate(target_names)
    }
    rankings: Dict[str, list[Dict[str, Any]]] = {name: [] for name in target_names}
    for layer, head, transformed_predictions in results:
        raw_predictions = _inverse_targets(transformed_predictions, target_transform)
        for target_index, name in enumerate(target_names):
            metrics = _metric_row(
                target_values[:, target_index],
                raw_predictions[:, target_index],
                high_threshold=thresholds[name],
            )
            rankings[name].append({
                "layer": int(layer),
                "head": int(head),
                "oof_spearman": metrics["spearman"],
                "oof_pearson": metrics["pearson"],
                "oof_r2": metrics["r2"],
                "oof_mae": metrics["mae"],
                "oof_high_auroc": metrics["high_auroc"],
            })
    for name in target_names:
        rankings[name].sort(
            key=lambda row: (
                -math.inf if not np.isfinite(row["oof_spearman"]) else row["oof_spearman"],
                -math.inf if not np.isfinite(row["oof_high_auroc"]) else row["oof_high_auroc"],
            ),
            reverse=True,
        )
    return rankings, thresholds, folds


def _fit_selected_models(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray | None,
    *,
    selected_pairs: Sequence[tuple[int, int]],
    target_transform: str,
    ridge_alpha: float,
) -> tuple[Dict[tuple[int, int], Dict[str, Any]], Dict[tuple[int, int], np.ndarray]]:
    transformed = _transform_targets(train_targets, target_transform)
    models: Dict[tuple[int, int], Dict[str, Any]] = {}
    predictions: Dict[tuple[int, int], np.ndarray] = {}
    for layer, head in selected_pairs:
        train_x = np.asarray(train_features[:, layer, head], dtype=np.float64)
        scaler = StandardScaler().fit(train_x)
        model = Ridge(alpha=float(ridge_alpha)).fit(
            scaler.transform(train_x), transformed
        )
        models[(layer, head)] = {
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "coef": np.asarray(model.coef_, dtype=np.float32),
            "intercept": np.asarray(model.intercept_, dtype=np.float32),
        }
        if validation_features is not None:
            validation_x = np.asarray(
                validation_features[:, layer, head], dtype=np.float64
            )
            prediction = model.predict(scaler.transform(validation_x))
            predictions[(layer, head)] = _inverse_targets(
                prediction, target_transform
            )
    return models, predictions


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def _richer_view(left: float | None, right: float | None) -> str | None:
    if left is None or right is None:
        return None
    if not np.isfinite(left) or not np.isfinite(right):
        return None
    if np.isclose(left, right):
        return "tie"
    return "greedy_answer" if right > left else "answer_free"


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    train_manifest = _load_manifest(features_dir, args.train_split)
    train_ids, train_groups, train_targets = _metadata_arrays(
        train_manifest, args.targets
    )
    original_train_samples = int(len(train_ids))
    excluded_train_samples = 0
    excluded_overlap_groups: list[str] = []
    validation_manifest = None
    validation_ids = validation_groups = validation_targets = None
    validation_path = features_dir / args.validation_split / "manifest.json"
    if validation_path.exists():
        validation_manifest = _load_manifest(features_dir, args.validation_split)
        validation_ids, validation_groups, validation_targets = _metadata_arrays(
            validation_manifest, args.targets
        )
        overlap = set(train_ids).intersection(set(validation_ids))
        if overlap:
            raise ValueError(
                f"Train/validation sample IDs overlap: {sorted(overlap)[:5]}"
            )
        group_overlap = set(train_groups).intersection(set(validation_groups))
        if group_overlap:
            if not args.exclude_validation_group_overlap:
                raise ValueError(
                    "Train/validation image groups overlap, which would leak image "
                    f"content into held-out evaluation: {sorted(group_overlap)[:5]}"
                )
            excluded_overlap_groups = sorted(str(value) for value in group_overlap)
            filtered_records = [
                row for row in train_manifest["records"]
                if str(row.get("group_id", row["sample_id"]))
                not in set(excluded_overlap_groups)
            ]
            excluded_train_samples = len(train_manifest["records"]) - len(filtered_records)
            if not filtered_records:
                raise ValueError("Removing validation-overlap groups emptied train split.")
            train_manifest = {**train_manifest, "records": filtered_records}
            train_ids, train_groups, train_targets = _metadata_arrays(
                train_manifest, args.targets
            )

    all_ranking_rows = []
    selected_rows = []
    comparison_rows = []
    artifact: Dict[str, Any] = {
        "method": "Qwen per-head LGD uncertainty Ridge probes",
        "version": 1,
        "targets": list(args.targets),
        "views": list(args.views),
        "train_sample_ids": train_ids.tolist(),
        "validation_sample_ids": (
            validation_ids.tolist() if validation_ids is not None else []
        ),
        "config": {
            "features_dir": str(features_dir.expanduser().resolve()),
            "num_folds": args.num_folds,
            "ridge_alpha": args.ridge_alpha,
            "high_quantile": args.high_quantile,
            "target_transform": args.target_transform,
            "top_heads": args.top_heads,
            "exclude_validation_group_overlap": bool(
                args.exclude_validation_group_overlap
            ),
            "original_train_samples": original_train_samples,
            "excluded_train_samples": excluded_train_samples,
            "excluded_overlap_groups": excluded_overlap_groups,
        },
        "views_artifact": {},
    }
    summary: Dict[str, Any] = {
        "method": artifact["method"],
        "train_samples": int(len(train_ids)),
        "original_train_samples": original_train_samples,
        "excluded_train_samples": excluded_train_samples,
        "excluded_overlap_group_count": len(excluded_overlap_groups),
        "train_image_groups": int(np.unique(train_groups).size),
        "validation_samples": int(len(validation_ids)) if validation_ids is not None else 0,
        "validation_image_groups": (
            int(np.unique(validation_groups).size)
            if validation_groups is not None else 0
        ),
        "targets": list(args.targets),
        "views": {},
    }

    for view in args.views:
        train_features = load_view_features(
            features_dir, train_manifest, split=args.train_split, view=view
        )
        validation_features = (
            load_view_features(
                features_dir,
                validation_manifest,
                split=args.validation_split,
                view=view,
            )
            if validation_manifest is not None else None
        )
        rankings, thresholds, folds = fit_oof_head_probes(
            train_features,
            train_targets,
            train_groups,
            target_names=args.targets,
            num_folds=args.num_folds,
            ridge_alpha=args.ridge_alpha,
            high_quantile=args.high_quantile,
            target_transform=args.target_transform,
            jobs=args.jobs,
        )
        model_layer_ids = list(
            train_manifest.get("model_layer_ids", range(train_features.shape[1]))
        )
        if len(model_layer_ids) != train_features.shape[1]:
            raise ValueError("model_layer_ids must align with the captured layer axis.")
        for ranking in rankings.values():
            for row in ranking:
                row["model_layer"] = int(model_layer_ids[int(row["layer"])])
        selected_by_target = {
            target: [
                (int(row["layer"]), int(row["head"]))
                for row in rankings[target][: int(args.top_heads)]
            ]
            for target in args.targets
        }
        selected_union = sorted({
            pair for pairs in selected_by_target.values() for pair in pairs
        })
        models, validation_predictions = _fit_selected_models(
            train_features,
            train_targets,
            validation_features,
            selected_pairs=selected_union,
            target_transform=args.target_transform,
            ridge_alpha=args.ridge_alpha,
        )
        fold_overlap = [
            int(len(set(train_groups[train]).intersection(set(train_groups[val]))))
            for train, val in folds
        ]
        view_summary: Dict[str, Any] = {
            "feature_shape": list(train_features.shape[1:]),
            "high_uncertainty_thresholds": thresholds,
            "train_high_uncertainty_positive_rates": {
                target: float(np.mean(
                    train_targets[:, target_index] >= thresholds[target]
                ))
                for target_index, target in enumerate(args.targets)
            },
            "validation_high_uncertainty_positive_rates": (
                {
                    target: float(np.mean(
                        validation_targets[:, target_index] >= thresholds[target]
                    ))
                    for target_index, target in enumerate(args.targets)
                }
                if validation_targets is not None else None
            ),
            "fold_group_overlap": fold_overlap,
            "targets": {},
        }
        for target_index, target in enumerate(args.targets):
            ranking = rankings[target]
            for rank, row in enumerate(ranking, 1):
                all_ranking_rows.append({
                    "view": view,
                    "target": target,
                    "rank": rank,
                    **row,
                })
            target_selected_rows = []
            for rank, base in enumerate(ranking[: int(args.top_heads)], 1):
                pair = (int(base["layer"]), int(base["head"]))
                row = {
                    "view": view,
                    "target": target,
                    "rank": rank,
                    **base,
                    "validation_spearman": None,
                    "validation_pearson": None,
                    "validation_r2": None,
                    "validation_mae": None,
                    "validation_high_auroc": None,
                }
                if validation_targets is not None:
                    metrics = _metric_row(
                        validation_targets[:, target_index],
                        validation_predictions[pair][:, target_index],
                        high_threshold=thresholds[target],
                    )
                    row.update({
                        "validation_spearman": metrics["spearman"],
                        "validation_pearson": metrics["pearson"],
                        "validation_r2": metrics["r2"],
                        "validation_mae": metrics["mae"],
                        "validation_high_auroc": metrics["high_auroc"],
                    })
                target_selected_rows.append(row)
                selected_rows.append(row)
            view_summary["targets"][target] = {
                "selected_heads": [list(pair) for pair in selected_by_target[target]],
                "best_discovery_head": target_selected_rows[0] if target_selected_rows else None,
                "train_selected_best_validation_spearman": (
                    target_selected_rows[0]["validation_spearman"]
                    if target_selected_rows and validation_targets is not None else None
                ),
                "train_selected_best_validation_high_auroc": (
                    target_selected_rows[0]["validation_high_auroc"]
                    if target_selected_rows and validation_targets is not None else None
                ),
                "selected_median_oof_spearman": _median(
                    target_selected_rows, "oof_spearman"
                ),
                "selected_median_oof_high_auroc": _median(
                    target_selected_rows, "oof_high_auroc"
                ),
                "selected_median_validation_spearman": _median(
                    target_selected_rows, "validation_spearman"
                ) if validation_targets is not None else None,
                "selected_median_validation_high_auroc": _median(
                    target_selected_rows, "validation_high_auroc"
                ) if validation_targets is not None else None,
            }
        summary["views"][view] = view_summary
        artifact["views_artifact"][view] = {
            "feature_shape": list(train_features.shape[1:]),
            "model_layer_ids": model_layer_ids,
            "head_family": train_manifest.get(
                "head_family", "full_attention_pre_o_proj"
            ),
            "rankings": rankings,
            "selected_heads": {
                target: [list(pair) for pair in pairs]
                for target, pairs in selected_by_target.items()
            },
            "high_uncertainty_thresholds": thresholds,
            "models": {
                f"L{layer}H{head}": model
                for (layer, head), model in models.items()
            },
        }
        del train_features, validation_features

    if set(DEFAULT_VIEWS).issubset(summary["views"]):
        for target in args.targets:
            answer_free = summary["views"]["answer_free"]["targets"][target]
            greedy = summary["views"]["greedy_answer"]["targets"][target]
            for metric in (
                "train_selected_best_validation_spearman",
                "train_selected_best_validation_high_auroc",
                "selected_median_oof_spearman",
                "selected_median_oof_high_auroc",
                "selected_median_validation_spearman",
                "selected_median_validation_high_auroc",
            ):
                left, right = answer_free.get(metric), greedy.get(metric)
                comparison_rows.append({
                    "target": target,
                    "metric": metric,
                    "answer_free": left,
                    "greedy_answer": right,
                    "greedy_minus_answer_free": (
                        float(right - left)
                        if left is not None and right is not None else None
                    ),
                })
            comparison_metric = (
                "train_selected_best_validation_spearman"
                if validation_targets is not None
                else "selected_median_oof_spearman"
            )
            answer_free_score = answer_free[comparison_metric]
            greedy_score = greedy[comparison_metric]
            summary.setdefault("view_comparison", {})[target] = {
                "primary_metric": comparison_metric,
                "answer_free": answer_free_score,
                "greedy_answer": greedy_score,
                "richer_view": _richer_view(answer_free_score, greedy_score),
                "greedy_minus_answer_free": (
                    float(greedy_score - answer_free_score)
                    if answer_free_score is not None and greedy_score is not None
                    and np.isfinite(answer_free_score) and np.isfinite(greedy_score)
                    else None
                ),
            }

    _write_csv(output_dir / "head_ranking.csv", all_ranking_rows)
    _write_csv(output_dir / "selected_head_validation.csv", selected_rows)
    _write_csv(output_dir / "view_comparison.csv", comparison_rows)
    _atomic_pickle(output_dir / "uncertainty_head_probes.pkl", artifact)
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path,
        required=True,
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--views", nargs="+", choices=DEFAULT_VIEWS, default=DEFAULT_VIEWS)
    parser.add_argument(
        "--targets", nargs="+", choices=AVAILABLE_TARGETS, default=TARGETS
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--target-transform", choices=("log1p", "none"), default="log1p")
    parser.add_argument("--top-heads", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--exclude-validation-group-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Remove train records whose image group occurs in validation instead "
            "of rejecting the dataset. The exclusion is saved in the artifact."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        required=True,
    )
    return parser


def main(args: argparse.Namespace) -> Dict[str, Any]:
    if args.num_folds < 2:
        raise ValueError("num_folds must be at least two.")
    if args.ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative.")
    if not 0 < args.high_quantile < 1:
        raise ValueError("high_quantile must lie in (0, 1).")
    if args.top_heads < 1:
        raise ValueError("top_heads must be positive.")
    if args.jobs == 0:
        raise ValueError("jobs cannot be zero.")
    args.views = tuple(dict.fromkeys(args.views))
    args.targets = tuple(dict.fromkeys(args.targets))
    return analyze(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
