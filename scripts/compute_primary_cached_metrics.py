#!/usr/bin/env python3
"""Add CoE/SIVR/trajectory scores to local compute results, without VLM calls.

SIVR uses the existing sequence classifier with 1500 train -> 500 validation,
fixed epochs and train-only normalization (no pooled cross-validation).
"""
import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def trajectory_features(record):
    trajectory = record["most_likely_answer"]["probe_features"]["primary_answer_trajectory"]
    sivr = trajectory["sivr"]
    sequence = np.stack([
        sivr["circular_variance_per_token"],
        sivr["covariance_logdet_mean_per_token"],
        sivr["output_entropy_per_token"],
    ], axis=1).astype(np.float32)
    if sequence.ndim != 2 or sequence.shape[0] == 0 or not np.isfinite(sequence).all():
        raise ValueError("Missing/nonfinite SIVR token sequence; regenerate using trajectory schema v2.")
    # Match uncertainty orientations in analyze_qwen_internal_trajectory.py.
    scores = {
        "primary_sivr_circular_variance": -float(sivr["circular_variance_mean"]),
        "primary_sivr_covariance_logdet": -float(sivr["covariance_logdet_mean"]),
        "primary_sivr_covariance_logdet_paper": -float(sivr["covariance_logdet_paper_mean"]),
    }
    for pool in ("mean", "last"):
        coe = trajectory[f"coe_{pool}"]
        for name in ("coe_r", "coe_c_repository", "coe_c_paper"):
            scores[f"primary_{pool}_{name}"] = -float(coe[name])
    for name in ("vocab_entropy", "hidden_channel_energy_entropy", "head_output_energy_entropy",
                 "jsd_to_final", "kl_final_to_layer", "adjacent_jsd", "hidden_delta_norm"):
        array = np.asarray(trajectory[name], dtype=np.float32)
        # Adjacent-layer changes have an intentionally undefined first row.
        finite = array[np.isfinite(array)]
        if not finite.size:
            raise ValueError(f"No defined values for {name}")
        scores[f"primary_{name}_mean"] = float(finite.mean())
    scores["primary_final_token_entropy"] = float(sequence[:, 2].mean())
    scores["primary_hidden_distance_to_final"] = float(
        1 - np.asarray(trajectory["hidden_cosine_to_final"], dtype=np.float32).mean())
    scores["primary_top1_flip_count"] = float(np.asarray(trajectory["top1_flip_count"]).mean())
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("Nonfinite primary trajectory score")
    return sequence, scores


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main(args):
    from compute_uncertainty_measures import _install_pil_pickle_compat
    from analyze_results import analyze_run
    from experiments.train_qwen_sivr_sequence_classifier import train_fold

    _install_pil_pickle_compat()
    source = args.run_dir.resolve()
    source = source / "files" if (source / "files").is_dir() else source
    output = args.output_dir.resolve()
    if output == source or output.is_relative_to(source) or output == args.base_results.resolve().parent:
        raise ValueError("Use a separate output directory; generation and base results are read-only inputs.")
    results = load_pickle(args.base_results)
    train = load_pickle(source / "train_generations.pkl")
    validation = load_pickle(source / "validation_generations.pkl")
    ids = results["validation_sample_ids"]
    records = [validation[sample_id] for sample_id in ids]
    truth = np.asarray([1 - float(r["most_likely_answer"]["accuracy"]) for r in records])
    if not np.array_equal(truth, np.asarray(results["validation_is_false"])):
        raise ValueError("Validation IDs/correctness do not match base compute results")
    if set(train).intersection(ids):
        raise ValueError("Train and validation sample IDs overlap")
    train_records = list(train.values())
    train_labels = np.asarray([1 - float(r["most_likely_answer"]["accuracy"]) for r in train_records])
    if np.unique(train_labels).size != 2:
        raise ValueError("SIVR classifier requires both correct and incorrect training answers")
    train_features = [trajectory_features(r) for r in train_records]
    validation_features = [trajectory_features(r) for r in records]
    all_sequences = [x[0] for x in train_features + validation_features]
    count = len(train_records)
    training_args = argparse.Namespace(
        epochs=args.sivr_epochs, batch_size=128, learning_rate=1e-4,
        weight_decay=1e-5, d_model=128, nhead=4, num_layers=2,
        dim_feedforward=256, dropout=0.1, pool="attn",
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"SIVR: train={count}, validation={len(ids)}, epochs={args.sivr_epochs}, device={device}", flush=True)
    predictions = train_fold(
        all_sequences, np.concatenate([train_labels, truth]),
        np.arange(count), np.arange(count, count + len(ids)),
        args=training_args, seed=args.seed, device=device,
    )
    measures = results.setdefault("uncertainty_measures", {})
    for name in validation_features[0][1]:
        measures[name] = [features[1][name] for features in validation_features]
    measures["sivr_sequence_error_probability"] = predictions.tolist()
    log_p_true = np.asarray([r["p_true"] for r in records], dtype=float)
    if not np.isfinite(log_p_true).all() or np.any(log_p_true > 1e-6):
        raise ValueError("p_true must contain finite log probabilities <= 0")
    results["p_true_probability"] = np.exp(log_p_true).tolist()
    measures["p_false_fixed"] = (1 - np.exp(log_p_true)).tolist()
    results["primary_cached_metrics_protocol"] = {
        "source_run": str(source), "base_results": str(args.base_results.resolve()),
        "train_count": count, "validation_count": len(ids),
        "sivr_training": vars(training_args), "seed": args.seed,
        "sivr_features": ["circular_variance", "covariance_logdet_mean", "final_layer_entropy"],
        "normalization": "train_tokens_only", "model_selection": "fixed_epochs_no_validation_selection",
        "answer_protocol": "saved_primary_answer_exact_tokens_causal_predictive_queries",
        "p_true_protocol": "legacy_text_only_self_evaluation",
        "scalar_signs": "COE and SIVR dispersion signs match existing trajectory analyzer",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "uncertainty_measures.pkl").open("wb") as handle:
        pickle.dump(results, handle)
    (output / "protocol.json").write_text(json.dumps(results["primary_cached_metrics_protocol"], indent=2) + "\n")
    names = [*validation_features[0][1], "sivr_sequence_error_probability"]
    with (output / "primary_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "is_false", "p_true", "p_false", *names])
        writer.writeheader()
        for index, sample_id in enumerate(ids):
            writer.writerow({"sample_id": sample_id, "is_false": truth[index],
                             "p_true": np.exp(log_p_true[index]), "p_false": measures["p_false_fixed"][index],
                             **{name: measures[name][index] for name in names}})
    analyze_run(None, local_run_dir=str(output))
    print(f"Combined metrics: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sivr-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.sivr_epochs < 1:
        parser.error("--sivr-epochs must be positive")
    main(args)
