#!/usr/bin/env python3
"""Select strong LGD auxiliary models on train and recompute all U labels.

Selection never reads validation performance.  Auxiliary and target models are
compared under the same stored multi-draw protocol using grounding
accuracy@0.5.  A model must also satisfy parsing and sample-coverage gates.
If fewer than ``--min-models`` pass the near-target gate, the best remaining
models may only be used when they satisfy the explicit hard maximum gap; if
that is still insufficient, the script fails instead of admitting weak
teachers.

All labels are recomputed from cached parsed boxes, so no model/API is called.
The compact ``label_sidecar.json`` can be passed to
``collect_qwen_uncertainty_heads.py --label-sidecar``.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from ceor.io import json_safe
from sklearn.metrics import roc_auc_score

from uncertainty.uncertainty_measures.lgd_uq import (
    LGDUQConfig,
    box_iou,
    recover_lgd_uncertainty,
)


LOGGER = logging.getLogger("lgd-quality-filtered-labels")
TARGETS = ("u_a", "u_d", "u_e", "u_total")


def _files_dir(run_dir: Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    return run_dir / "files" if (run_dir / "files").is_dir() else run_dir


def _install_pil_pickle_compat() -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    if getattr(Image.Image, "_semunc_pickle_compat", False):
        return
    original = Image.Image.__setstate__

    def compat(image, state):
        if isinstance(state, (tuple, list)) and len(state) > 5:
            state = state[:5]
        return original(image, state)

    Image.Image.__setstate__ = compat
    Image.Image._semunc_pickle_compat = True


def _load_pickle(path: Path) -> Any:
    _install_pil_pickle_compat()
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _as_box(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != 4 or not np.isfinite(array).all():
        return None
    x1, y1, x2, y2 = map(float, array)
    if min(x1, y1, x2, y2) < 0.0 or max(x1, y1, x2, y2) > 1.0:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _new_stats() -> dict[str, float]:
    return {
        "covered_samples": 0,
        "draws": 0,
        "valid_draws": 0,
        "correct_draws": 0,
        "iou_sum": 0.0,
    }


def _update_draw_stats(
    stats: dict[str, float],
    parsed_boxes: Sequence[Any],
    ground_truth: tuple[float, float, float, float],
) -> None:
    stats["covered_samples"] += 1
    for value in parsed_boxes:
        stats["draws"] += 1
        box = _as_box(value)
        if box is None:
            continue
        overlap = box_iou(ground_truth, box)
        stats["valid_draws"] += 1
        stats["correct_draws"] += float(overlap >= 0.5)
        stats["iou_sum"] += overlap


def _training_quality(
    generations: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model_stats: dict[str, dict[str, float]] = {}
    target_stats = _new_stats()
    greedy_total = greedy_correct = greedy_iou_sum = 0.0
    usable_samples = 0
    for record in generations.values():
        lgd = record.get("lgd_uq")
        if not isinstance(lgd, Mapping):
            continue
        diagnostics = lgd.get("diagnostics", {})
        ground_truth = _as_box(diagnostics.get("ground_truth_box"))
        auxiliary = lgd.get("auxiliary_models")
        target = lgd.get("target_model")
        if ground_truth is None or not isinstance(auxiliary, Mapping):
            continue
        usable_samples += 1
        for model_name, model_record in auxiliary.items():
            if not isinstance(model_record, Mapping):
                continue
            stats = model_stats.setdefault(str(model_name), _new_stats())
            _update_draw_stats(stats, model_record.get("parsed_boxes", []), ground_truth)
        if isinstance(target, Mapping):
            _update_draw_stats(target_stats, target.get("parsed_boxes", []), ground_truth)
        most_likely = record.get("most_likely_answer", {})
        grounding = (
            most_likely.get("grounding_eval", {})
            if isinstance(most_likely, Mapping) else {}
        )
        greedy_iou = float(grounding.get("iou", 0.0))
        greedy_total += 1
        greedy_correct += float(greedy_iou >= 0.5)
        greedy_iou_sum += greedy_iou
    if usable_samples == 0:
        raise ValueError("No train records with embedded LGD parsed boxes were found.")

    def finalize(name: str, stats: Mapping[str, float]) -> dict[str, Any]:
        draws = max(float(stats["draws"]), 1.0)
        return {
            "model": name,
            "covered_samples": int(stats["covered_samples"]),
            "sample_coverage": float(stats["covered_samples"] / usable_samples),
            "draws": int(stats["draws"]),
            "valid_draws": int(stats["valid_draws"]),
            "parse_rate": float(stats["valid_draws"] / draws),
            "accuracy_at_0_5": float(stats["correct_draws"] / draws),
            "mean_iou_all_draws": float(stats["iou_sum"] / draws),
        }

    rows = [finalize(name, stats) for name, stats in sorted(model_stats.items())]
    target_row = finalize("__target_sampled__", target_stats)
    target_row.update({
        "greedy_accuracy_at_0_5": float(greedy_correct / max(greedy_total, 1.0)),
        "greedy_mean_iou": float(greedy_iou_sum / max(greedy_total, 1.0)),
        "num_train_records": int(usable_samples),
    })
    return rows, target_row


def select_models(
    quality_rows: Sequence[Mapping[str, Any]],
    *,
    target_accuracy: float,
    relative_tolerance: float,
    fallback_max_gap: float,
    min_accuracy: float,
    min_parse_rate: float,
    min_sample_coverage: float,
    min_models: int,
) -> tuple[list[str], dict[str, Any]]:
    near_cutoff = max(float(min_accuracy), float(target_accuracy) - float(relative_tolerance))
    hard_cutoff = max(float(min_accuracy), float(target_accuracy) - float(fallback_max_gap))
    valid_quality = [
        row for row in quality_rows
        if float(row["parse_rate"]) >= min_parse_rate
        and float(row["sample_coverage"]) >= min_sample_coverage
    ]
    near = [row for row in valid_quality if float(row["accuracy_at_0_5"]) >= near_cutoff]
    selected = sorted(near, key=lambda row: float(row["accuracy_at_0_5"]), reverse=True)
    fallback_used: list[str] = []
    if len(selected) < min_models:
        candidates = sorted(
            (
                row for row in valid_quality
                if row not in selected and float(row["accuracy_at_0_5"]) >= hard_cutoff
            ),
            key=lambda row: float(row["accuracy_at_0_5"]),
            reverse=True,
        )
        needed = min_models - len(selected)
        selected.extend(candidates[:needed])
        fallback_used = [str(row["model"]) for row in candidates[:needed]]
    if len(selected) < min_models:
        ranked = sorted(
            quality_rows,
            key=lambda row: float(row["accuracy_at_0_5"]),
            reverse=True,
        )
        detail = ", ".join(
            f"{row['model']}={float(row['accuracy_at_0_5']):.3f}"
            for row in ranked
        )
        raise ValueError(
            f"Only {len(selected)} auxiliary models satisfy the hard quality gates; "
            f"need at least {min_models}. target={target_accuracy:.3f}, "
            f"near_cutoff={near_cutoff:.3f}, hard_cutoff={hard_cutoff:.3f}. "
            f"Available: {detail}. Add stronger auxiliary models or explicitly "
            "relax --fallback-max-gap."
        )
    names = [str(row["model"]) for row in selected]
    audit = {
        "target_accuracy": float(target_accuracy),
        "near_cutoff": near_cutoff,
        "hard_cutoff": hard_cutoff,
        "min_accuracy": float(min_accuracy),
        "min_parse_rate": float(min_parse_rate),
        "min_sample_coverage": float(min_sample_coverage),
        "min_models": int(min_models),
        "fallback_models": fallback_used,
        "selected_models": names,
    }
    return names, audit


def _lgd_config(lgd: Mapping[str, Any]) -> LGDUQConfig:
    raw = lgd.get("config", {})
    return LGDUQConfig(
        iou_threshold=float(raw.get("iou_threshold", 0.5)),
        smoothing=float(raw.get("smoothing", 0.01)),
        qwen_coordinate_scale=float(raw.get("qwen_coordinate_scale", 999.0)),
    )


def _recompute_split(
    generations: Mapping[str, Mapping[str, Any]],
    selected_models: Sequence[str],
    *,
    min_models_per_sample: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records: dict[str, Any] = {}
    original_total: list[float] = []
    filtered_total: list[float] = []
    errors: list[bool] = []
    available_counts: list[int] = []
    for sample_id, record in generations.items():
        lgd = record.get("lgd_uq")
        if not isinstance(lgd, Mapping):
            raise ValueError(f"Sample {sample_id!r} has no embedded LGD result.")
        auxiliary = lgd.get("auxiliary_models", {})
        target = lgd.get("target_model")
        if not isinstance(auxiliary, Mapping) or not isinstance(target, Mapping):
            raise ValueError(f"Sample {sample_id!r} lacks LGD model records.")
        present = [name for name in selected_models if name in auxiliary]
        if len(present) < min_models_per_sample:
            raise ValueError(
                f"Sample {sample_id!r} has only {len(present)} selected auxiliary "
                "models, below "
                f"--min-models-per-sample={min_models_per_sample}."
            )
        predictions = {
            name: auxiliary[name].get("parsed_boxes", []) for name in present
        }
        weights = {
            name: float(auxiliary[name].get("weight", 1.0)) for name in present
        }
        recomputed = recover_lgd_uncertainty(
            ground_truth_box=lgd["diagnostics"]["ground_truth_box"],
            auxiliary_predictions=predictions,
            target_predictions=target.get("parsed_boxes", []),
            auxiliary_weights=weights,
            config=_lgd_config(lgd),
        )
        uncertainty = recomputed["uncertainty"]
        labels = {name: float(uncertainty[name]) for name in TARGETS}
        records[str(sample_id)] = {
            "labels": labels,
            "available_selected_models": present,
            "num_auxiliary_models": len(present),
        }
        original_total.append(float(lgd["uncertainty"]["u_total"]))
        filtered_total.append(labels["u_total"])
        most_likely = record.get("most_likely_answer", {})
        grounding = (
            most_likely.get("grounding_eval", {})
            if isinstance(most_likely, Mapping) else {}
        )
        errors.append(float(grounding.get("iou", 0.0)) < 0.5)
        available_counts.append(len(present))
    labels_array = np.asarray(errors, dtype=np.int64)

    def auc(values: Sequence[float]) -> float:
        return (
            float(roc_auc_score(labels_array, values))
            if np.unique(labels_array).size == 2 else float("nan")
        )

    audit = {
        "num_samples": len(records),
        "num_errors_iou_lt_0_5": int(labels_array.sum()),
        "min_available_selected_models": int(min(available_counts)),
        "max_available_selected_models": int(max(available_counts)),
        "original_u_total_error_auroc": auc(original_total),
        "filtered_u_total_error_auroc": auc(filtered_total),
    }
    return records, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    files_dir = _files_dir(args.run_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = files_dir / "train_generations.pkl"
    validation_path = files_dir / "validation_generations.pkl"
    LOGGER.info("Loading train generations: %s", train_path)
    train = _load_pickle(train_path)
    quality_rows, target_quality = _training_quality(train)
    target_accuracy = (
        float(target_quality["accuracy_at_0_5"])
        if args.target_reference == "sampled"
        else float(target_quality["greedy_accuracy_at_0_5"])
    )
    selected, selection = select_models(
        quality_rows,
        target_accuracy=target_accuracy,
        relative_tolerance=args.relative_tolerance,
        fallback_max_gap=args.fallback_max_gap,
        min_accuracy=args.min_accuracy,
        min_parse_rate=args.min_parse_rate,
        min_sample_coverage=args.min_sample_coverage,
        min_models=args.min_models,
    )
    quality_output = []
    for row in quality_rows:
        quality_output.append({
            **row,
            "target_reference": args.target_reference,
            "target_accuracy": target_accuracy,
            "accuracy_gap_vs_target": float(row["accuracy_at_0_5"]) - target_accuracy,
            "selected": str(row["model"]) in selected,
            "selection_mode": (
                "fallback" if str(row["model"]) in selection["fallback_models"]
                else "near_target" if str(row["model"]) in selected else "excluded"
            ),
        })
    _write_csv(output_dir / "model_quality_train.csv", quality_output)
    LOGGER.info("Selected auxiliary models: %s", ", ".join(selected))
    train_labels, train_audit = _recompute_split(
        train, selected, min_models_per_sample=args.min_models_per_sample
    )
    del train
    LOGGER.info("Loading validation generations: %s", validation_path)
    validation = _load_pickle(validation_path)
    validation_labels, validation_audit = _recompute_split(
        validation, selected, min_models_per_sample=args.min_models_per_sample
    )
    del validation
    sidecar = {
        "schema_version": 1,
        "method": "train-only quality-filtered LGD auxiliary ensemble",
        "source_run_files": str(files_dir),
        "selection_split": "train",
        "selection": {
            **selection,
            "target_reference": args.target_reference,
            "relative_tolerance": float(args.relative_tolerance),
            "fallback_max_gap": float(args.fallback_max_gap),
            "min_models_per_sample": int(args.min_models_per_sample),
        },
        "target_quality_train": target_quality,
        "model_quality_train": quality_output,
        "split_audit": {
            "train": train_audit,
            "validation": validation_audit,
        },
        "records": {
            "train": train_labels,
            "validation": validation_labels,
        },
    }
    _write_json(output_dir / "label_sidecar.json", sidecar)
    _write_json(output_dir / "selection_summary.json", {
        key: value for key, value in sidecar.items() if key != "records"
    })
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-reference", choices=("sampled", "greedy"), default="sampled",
        help="Use matched multi-draw target accuracy by default.",
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--fallback-max-gap", type=float, default=0.15)
    parser.add_argument("--min-accuracy", type=float, default=0.20)
    parser.add_argument("--min-parse-rate", type=float, default=0.90)
    parser.add_argument("--min-sample-coverage", type=float, default=0.95)
    parser.add_argument("--min-models", type=int, default=3)
    parser.add_argument(
        "--min-models-per-sample", type=int, default=2,
        help=(
            "Minimum available selected teachers after rare provider-side filtering; "
            "global selection still obeys --min-models."
        ),
    )
    return parser


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in (
        "relative_tolerance", "fallback_max_gap", "min_accuracy",
        "min_parse_rate", "min_sample_coverage",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1].")
    if args.relative_tolerance > args.fallback_max_gap:
        parser.error("--relative-tolerance cannot exceed --fallback-max-gap.")
    if args.min_models < 2:
        parser.error("--min-models must be at least 2; 3 is recommended.")
    if not 2 <= args.min_models_per_sample <= args.min_models:
        parser.error(
            "--min-models-per-sample must be at least 2 and no larger than --min-models."
        )
    run(args)


if __name__ == "__main__":
    main()
