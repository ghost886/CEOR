"""Lightweight hidden-state probes for generated answers."""
import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def feature_matrix(
    probe_features: Iterable[Dict],
    feature_name: str = "argus_h_mean",
) -> np.ndarray:
    """Convert one saved probe feature into a dense sample matrix."""
    rows = []
    for features in probe_features:
        value = features.get(feature_name)
        if value is None:
            raise ValueError(f"Probe feature `{feature_name}` is missing.")
        if torch.is_tensor(value):
            value = value.float().cpu().numpy()
        rows.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.stack(rows, axis=0)


def fit_sep_and_score(
    train_features: List[Dict],
    train_is_false: List[float],
    eval_features: List[Dict],
    *,
    feature_name: str = "argus_h_mean",
) -> Tuple[np.ndarray, object]:
    """Train a linear SEP probe and return eval error probabilities."""
    x_train = feature_matrix(train_features, feature_name)
    x_eval = feature_matrix(eval_features, feature_name)
    y_train = np.asarray(train_is_false, dtype=np.int64)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
        ),
    )
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_eval)[:, 1]
    logging.info("SEP probe trained on %d samples with %d dims.", *x_train.shape)
    return scores, clf
