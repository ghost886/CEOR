"""Source-training grouped Ridge probes and Eq. (11) layer selection."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ceor.io import atomic_json
from ceor.occlusion import layer_ids_for_shape


def select_key_layer(probe):
    view = probe["views_artifact"]["answer_free"]
    shape = list(view["feature_shape"])
    ids = layer_ids_for_shape(shape, view.get("model_layer_ids"))
    scores = np.full(shape[:2], np.nan)
    seen = set()
    for row in view["rankings"]["u_total"]:
        layer, head = int(row["layer"]), int(row["head"])
        if (layer, head) in seen or not (0 <= layer < shape[0] and 0 <= head < shape[1]):
            raise ValueError("Invalid or duplicate head in the OOF rankings.")
        seen.add((layer, head))
        scores[layer, head] = float(row["oof_spearman"])
    if len(seen) != shape[0] * shape[1]:
        raise ValueError("Key1 selection requires OOF scores for every head.")
    # Undefined correlations make the layer ineligible. Never average only a
    # selected or finite subset of heads, which would change manuscript Eq. 11.
    means = scores.mean(axis=1)
    eligible = np.flatnonzero(np.isfinite(means))
    if not len(eligible):
        raise ValueError("No layer has defined correlations for all heads.")
    order = eligible[np.argsort(-means[eligible], kind="stable")]
    key_index = int(order[0])
    return {
        "schema_version": 1,
        "method": "mean_all_head_grouped_train_oof_spearman_u_total",
        "key_layer": ids[key_index],
        "key_layer_index": key_index,
        "feature_shape": shape,
        "model_layer_ids": ids,
        "ranked_layers": [ids[int(i)] for i in order],
        "mean_oof_spearman": [float(x) if np.isfinite(x) else None for x in means],
        "response_layers": [x for x in ids if x > ids[key_index]],
        "source_training_only": True,
        "probe_config": probe.get("config", {}),
    }


def fit_probes(args):
    from experiments.analyze_qwen_uncertainty_heads import AVAILABLE_TARGETS, main

    features_dir = Path(args.features_dir)
    train_manifest = json.loads((features_dir / "train/manifest.json").read_text())
    shape = train_manifest["feature_shape"]
    layer_ids_for_shape(shape, train_manifest.get("model_layer_ids"))
    main(SimpleNamespace(
        features_dir=features_dir, output_dir=Path(args.output_dir),
        train_split="train", validation_split="validation", views=("answer_free",),
        targets=tuple(AVAILABLE_TARGETS), num_folds=args.num_folds,
        ridge_alpha=args.ridge_alpha, high_quantile=0.75, target_transform="log1p",
        top_heads=int(shape[0]) * int(shape[1]), jobs=args.jobs,
        exclude_validation_group_overlap=args.exclude_validation_group_overlap,
    ))
    with (Path(args.output_dir) / "uncertainty_head_probes.pkl").open("rb") as handle:
        probe = pickle.load(handle)
    selection = select_key_layer(probe)
    identity = train_manifest.get("collection_identity", {})
    selection["model"] = identity.get("model")
    selection["processor_limits"] = identity.get("processor_limits", {})
    path = Path(args.output_dir) / "key_layer.json"
    atomic_json(path, selection)
    print(f"Selected physical layer L{selection['key_layer']}; saved {path}")
    return selection
