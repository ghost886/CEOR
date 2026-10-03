#!/usr/bin/env python3
"""Collect answer-free and greedy-answer Qwen head features for LGD uncertainty.

The primary ``answer_free`` view performs a single multimodal prompt prefill and
captures the final prompt query immediately before the first answer token.  It
does not call ``generate``.  The auxiliary ``greedy_answer`` view reuses the
already collected last-answer-token pre-``o_proj`` tensor from a local W&B run.

Both views therefore have exactly the same semantics and shape as the existing
Qwen VIB/ITI feature: ``[layers, heads, head_dim]`` before ``o_proj``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from ceor.io import image_digest
from uncertainty.models.qwen_vl_models import QwenVLModel
from uncertainty.uncertainty_measures.paper_probe_features import (
    AttentionHeadOutputCapture,
    VIB_FEATURE_NAME,
)
from uncertainty.utils import utils as uncertainty_utils


LOGGER = logging.getLogger("qwen-uncertainty-head-collection")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
VIEWS = ("answer_free", "greedy_answer")
TARGETS = ("u_a", "u_d", "u_e")
GENERATE_ANSWERS_GROUNDING_PROMPT = uncertainty_utils.get_make_prompt(
    SimpleNamespace(prompt_type="grounding")
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _files_dir(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    return run_dir / "files" if (run_dir / "files").is_dir() else run_dir


def grounding_prompt(question: str) -> str:
    """Use the exact grounding prompt builder used by ``generate_answers.py``."""
    return GENERATE_ANSWERS_GROUNDING_PROMPT(
        None,
        str(question).strip(),
        None,
        "",
        False,
    )


def _as_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            from io import BytesIO

            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path")
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as image:
            return image.convert("RGB")
    raise TypeError(f"Unsupported image representation: {type(value)!r}")


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def extract_source_metadata(
    sample_id: str,
    record: Mapping[str, Any],
    *,
    labels_override: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Extract labels and audit metadata without retaining the source image."""
    if labels_override is None:
        lgd = record.get("lgd_uq")
        uncertainty = lgd.get("uncertainty") if isinstance(lgd, Mapping) else None
        if not isinstance(uncertainty, Mapping):
            raise ValueError(f"Sample {sample_id!r} has no embedded LGD uncertainty.")
    else:
        uncertainty = labels_override
    labels = {
        name: _finite_float(uncertainty.get(name), f"{sample_id}.{name}")
        for name in (*TARGETS, "u_total")
    }
    most_likely = record.get("most_likely_answer")
    if not isinstance(most_likely, Mapping):
        raise ValueError(f"Sample {sample_id!r} has no target-model greedy answer.")
    response = str(most_likely.get("response", "")).strip()
    if not response:
        raise ValueError(f"Sample {sample_id!r} has an empty greedy answer.")
    sample_metadata = record.get("sample_metadata", {})
    if not isinstance(sample_metadata, Mapping):
        sample_metadata = {}
    grounding_eval = most_likely.get("grounding_eval", {})
    if not isinstance(grounding_eval, Mapping):
        grounding_eval = {}
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError(f"Sample {sample_id!r} has an empty question.")
    image_id = str(sample_metadata.get("image_id", sample_id))
    return {
        "sample_id": str(sample_id),
        "question": question,
        "prompt": most_likely.get("prompt") or grounding_prompt(question),
        "greedy_answer": response,
        "labels": labels,
        "group_id": image_id,
        "dataset_name": str(sample_metadata.get("dataset_name", "unknown")),
        "source_split": str(sample_metadata.get("split", "unknown")),
        "source_id": str(sample_metadata.get("source_id", "")),
        "image_id": image_id,
        "image_path": str(sample_metadata.get("image_path", "")),
        "greedy_iou": _finite_float(grounding_eval.get("iou", 0.0), "iou"),
        "greedy_correct": int(float(grounding_eval.get("accuracy", 0.0)) >= 0.5),
    }


def existing_greedy_head_feature(record: Mapping[str, Any]) -> torch.Tensor:
    """Read the source run's last-greedy-answer-token pre-o_proj feature."""
    most_likely = record.get("most_likely_answer")
    features = most_likely.get("probe_features") if isinstance(most_likely, Mapping) else None
    value = features.get(VIB_FEATURE_NAME) if isinstance(features, Mapping) else None
    if value is None:
        raise ValueError(
            f"Source greedy answer has no `{VIB_FEATURE_NAME}`. The W&B run must "
            "have been generated with probe-feature collection enabled."
        )
    tensor = torch.as_tensor(value).detach().float().cpu()
    if tensor.ndim != 3 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(
            f"Greedy head feature must be finite [layers, heads, head_dim], got "
            f"{tuple(tensor.shape)}."
        )
    return tensor.to(torch.float16)


def collect_answer_free_head_feature(
    model: QwenVLModel,
    *,
    prompt: str,
    image: Image.Image,
) -> Dict[str, Any]:
    """Capture the prompt-final query without generating an answer token.

    ``add_generation_prompt=True`` appends Qwen's fixed assistant-role prefix.
    The final prefill query is consequently the exact state whose logits would
    emit answer token zero.  Only a forward pass is performed.
    """
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image", "image": image},
    ]}]
    inputs = model.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.model.device)
    forward_kwargs = dict(inputs)
    forward_kwargs.update(use_cache=False, return_dict=True)
    model.model.eval()
    with AttentionHeadOutputCapture(model.model) as capture, torch.no_grad():
        try:
            outputs = model.model(**forward_kwargs, logits_to_keep=1)
        except TypeError:
            # Compatibility with transformer versions predating logits_to_keep.
            outputs = model.model(**forward_kwargs)
    feature = capture.final_tensor().to(torch.float16)
    next_token_id = None
    next_token_text = None
    logits = getattr(outputs, "logits", None)
    if torch.is_tensor(logits) and logits.numel():
        next_token_id = int(logits[0, -1].argmax().item())
        next_token_text = model.processor.decode(
            torch.tensor([next_token_id]), skip_special_tokens=False
        )
    result = {
        "feature": feature,
        "num_prompt_tokens": int(inputs["input_ids"].shape[1]),
        "predicted_first_token_id": next_token_id,
        "predicted_first_token_text": next_token_text,
        "generated_answer_tokens": 0,
    }
    del outputs, inputs, forward_kwargs
    return result


def _identity(args: argparse.Namespace, split: str, source_path: Path) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "Qwen sample-level uncertainty head localization",
        "split": str(split),
        "source_generations": str(source_path.resolve()),
        "model": str(args.model),
        "source_sha256": _sha256(source_path),
        "processor_limits": {
            key: os.getenv(key) for key in ("QWEN_VL_MIN_PIXELS", "QWEN_VL_MAX_PIXELS")
        },
        "views": list(args.views),
        "prompt_type": "grounding",
        "prompt_builder": "uncertainty.utils.utils.get_make_prompt",
        "max_samples": int(args.max_samples),
        "expected_dataset": str(args.expected_dataset),
        "label_sidecar": (
            {
                "path": str(Path(args.label_sidecar).resolve()),
                "sha256": str(args._label_sidecar_sha256),
            }
            if args.label_sidecar else None
        ),
        "reuse_features_dir": (
            str(Path(args.reuse_features_dir).resolve())
            if args.reuse_features_dir else None
        ),
        "answer_free_semantics": (
            "final multimodal prompt query before answer token zero; no generation"
        ),
        "greedy_answer_semantics": (
            "existing target-model greedy generation's final-query pre-o_proj feature"
        ),
    }


def _selected_items(
    generations: Mapping[Any, Any], max_samples: int
) -> Sequence[tuple[str, Mapping[str, Any]]]:
    items = [(str(key), value) for key, value in generations.items()]
    return items if int(max_samples) <= 0 else items[: int(max_samples)]


def _validate_or_write_identity(path: Path, identity: Mapping[str, Any]) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != identity:
            raise ValueError(
                f"Existing collection identity differs at {path}. Use another "
                "output directory or remove that split deliberately."
            )
    else:
        _atomic_json(path, identity)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_label_sidecar(path: Path | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("records"), Mapping):
        raise ValueError(f"Label sidecar has no split records: {path}")
    return payload


def _reuse_index(
    reuse_features_dir: Path | None,
    split: str,
) -> Dict[str, Mapping[str, Any]]:
    if reuse_features_dir is None:
        return {}
    manifest_path = Path(reuse_features_dir) / split / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        str(row["sample_id"]): row
        for row in manifest.get("records", [])
    }


def _reuse_manifest_metadata(
    reuse_features_dir: Path | None, split: str
) -> Dict[str, Any]:
    if reuse_features_dir is None:
        return {}
    path = Path(reuse_features_dir) / split / "manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        key: payload[key]
        for key in ("model_layer_ids", "head_family")
        if key in payload
    }


def _reuse_feature_file(
    *,
    reuse_features_dir: Path | None,
    split: str,
    reuse_row: Mapping[str, Any] | None,
    view: str,
    output_path: Path,
) -> bool:
    if reuse_features_dir is None or not isinstance(reuse_row, Mapping):
        return False
    relative = reuse_row.get("feature_files", {}).get(view)
    if not relative:
        return False
    source = Path(reuse_features_dir) / split / str(relative)
    if not source.exists():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, output_path)
    except OSError:
        shutil.copy2(source, output_path)
    return True


def collect_split(
    args: argparse.Namespace,
    *,
    split: str,
    model: QwenVLModel | None,
) -> tuple[Dict[str, Any], QwenVLModel | None]:
    files_dir = _files_dir(args.run_dir)
    source_path = files_dir / f"{split}_generations.pkl"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source artifact: {source_path}")
    split_dir = Path(args.output_dir) / split
    identity = _identity(args, split, source_path)
    _validate_or_write_identity(split_dir / "collection_identity.json", identity)

    LOGGER.info("Loading %s", source_path)
    with source_path.open("rb") as handle:
        generations = pickle.load(handle)
    if not isinstance(generations, Mapping):
        raise TypeError(f"{source_path} must contain a sample mapping.")
    items = _selected_items(generations, args.max_samples)
    reuse_rows = _reuse_index(args.reuse_features_dir, split)
    reuse_metadata = _reuse_manifest_metadata(args.reuse_features_dir, split)
    sidecar_split = (
        args._label_sidecar_payload.get("records", {}).get(split, {})
        if args._label_sidecar_payload is not None else None
    )
    if args._label_sidecar_payload is not None and not isinstance(sidecar_split, Mapping):
        raise ValueError(f"Label sidecar has no {split!r} mapping.")
    manifest_rows = []
    feature_shape = None

    for position, (sample_id, source_record) in enumerate(items):
        if not isinstance(source_record, Mapping):
            raise TypeError(f"Sample {sample_id!r} is not a mapping.")
        labels_override = None
        if sidecar_split is not None:
            label_record = sidecar_split.get(sample_id)
            if not isinstance(label_record, Mapping):
                raise ValueError(
                    f"Label sidecar has no {split}/{sample_id} record."
                )
            labels_override = label_record.get("labels", label_record)
        metadata = extract_source_metadata(
            sample_id,
            source_record,
            labels_override=labels_override,
        )
        metadata["label_source"] = (
            "quality_filtered_lgd_sidecar" if labels_override is not None
            else "embedded_lgd_uq"
        )
        if (
            args.expected_dataset
            and metadata["dataset_name"] != args.expected_dataset
        ):
            raise ValueError(
                f"Expected dataset {args.expected_dataset!r}, got "
                f"{metadata['dataset_name']!r} for {sample_id!r}."
            )
        stem = f"{position:06d}"
        feature_files: Dict[str, str] = {}
        view_shapes: Dict[str, Sequence[int]] = {}
        audit: Dict[str, Any] = {}
        reuse_row = reuse_rows.get(sample_id)

        if "greedy_answer" in args.views:
            relative = Path("greedy_answer") / f"{stem}.pt"
            output_path = split_dir / relative
            if not output_path.exists():
                reused = _reuse_feature_file(
                    reuse_features_dir=args.reuse_features_dir,
                    split=split,
                    reuse_row=reuse_row,
                    view="greedy_answer",
                    output_path=output_path,
                )
                if not reused:
                    feature = existing_greedy_head_feature(source_record)
                    _atomic_torch(output_path, {
                        "sample_id": sample_id,
                        "view": "greedy_answer",
                        "feature": feature,
                    })
                else:
                    saved = torch.load(
                        output_path, map_location="cpu", weights_only=False
                    )
                    feature = torch.as_tensor(saved["feature"])
            else:
                saved = torch.load(output_path, map_location="cpu", weights_only=False)
                feature = torch.as_tensor(saved["feature"])
            feature_files["greedy_answer"] = str(relative)
            view_shapes["greedy_answer"] = list(feature.shape)

        if "answer_free" in args.views:
            relative = Path("answer_free") / f"{stem}.pt"
            output_path = split_dir / relative
            if not output_path.exists():
                reused = _reuse_feature_file(
                    reuse_features_dir=args.reuse_features_dir,
                    split=split,
                    reuse_row=reuse_row,
                    view="answer_free",
                    output_path=output_path,
                )
                if not reused:
                    if model is None:
                        LOGGER.info("Loading target Qwen model for answer-free prefill.")
                        model = QwenVLModel(
                            args.model,
                            max_new_tokens=1,
                            stop_sequences="default",
                            collect_vib_probe=False,
                            collect_lrp_probe=False,
                        )
                    image = _as_image(source_record.get("image"))
                    collected = collect_answer_free_head_feature(
                        model, prompt=metadata["prompt"], image=image
                    )
                    feature = collected.pop("feature")
                    audit["answer_free"] = collected
                    _atomic_torch(output_path, {
                        "sample_id": sample_id,
                        "view": "answer_free",
                        "feature": feature,
                        "audit": collected,
                        "image_sha256": image_digest(image),
                    })
                else:
                    saved = torch.load(
                        output_path, map_location="cpu", weights_only=False
                    )
                    feature = torch.as_tensor(saved["feature"])
                    audit["answer_free"] = dict(saved.get("audit", {}))
            else:
                saved = torch.load(output_path, map_location="cpu", weights_only=False)
                feature = torch.as_tensor(saved["feature"])
                audit["answer_free"] = dict(saved.get("audit", {}))
            feature_files["answer_free"] = str(relative)
            view_shapes["answer_free"] = list(feature.shape)

        unique_shapes = {tuple(shape) for shape in view_shapes.values()}
        if len(unique_shapes) != 1:
            raise ValueError(
                f"Feature views disagree for {sample_id}: {view_shapes}."
            )
        current_shape = next(iter(unique_shapes))
        if len(current_shape) != 3:
            raise ValueError(f"Expected [layers, heads, head_dim], got {current_shape}.")
        if feature_shape is None:
            feature_shape = current_shape
        elif current_shape != feature_shape:
            raise ValueError(
                f"Feature shape changed from {feature_shape} to {current_shape}."
            )
        metadata.update({
            "source_position": int(position),
            "feature_files": feature_files,
            "feature_shapes": view_shapes,
            "audit": audit,
        })
        manifest_rows.append(metadata)
        LOGGER.info(
            "%s sample %d/%d: %s", split, position + 1, len(items), sample_id
        )
        if torch.cuda.is_available() and (position + 1) % int(args.empty_cache_every) == 0:
            torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "split": split,
        "num_samples": len(manifest_rows),
        "views": list(args.views),
        "targets": list(TARGETS),
        "feature_shape": list(feature_shape) if feature_shape is not None else None,
        **reuse_metadata,
        "collection_identity": identity,
        "records": manifest_rows,
    }
    _atomic_json(split_dir / "manifest.json", manifest)
    del generations
    gc.collect()
    return manifest, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument(
        "--views", nargs="+", choices=VIEWS, default=("answer_free",),
    )
    parser.add_argument(
        "--expected-dataset", default="",
        help="Reject a W&B run from another dataset; pass an empty string to disable.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Maximum per split; zero uses every source sample.",
    )
    parser.add_argument(
        "--label-sidecar", type=Path, default=None,
        help="Optional quality-filtered LGD labels produced by the preparation script.",
    )
    parser.add_argument(
        "--reuse-features-dir", type=Path, default=None,
        help="Reuse matching sample/view .pt files from an earlier collection.",
    )
    parser.add_argument("--empty-cache-every", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path,
        required=True,
    )
    return parser


def main(args: argparse.Namespace) -> Dict[str, Any]:
    if args.max_samples < 0:
        raise ValueError("max_samples must be non-negative.")
    if args.empty_cache_every < 1:
        raise ValueError("empty_cache_every must be positive.")
    args.views = tuple(dict.fromkeys(args.views))
    args.splits = tuple(dict.fromkeys(args.splits))
    args.label_sidecar = (
        Path(args.label_sidecar).expanduser().resolve()
        if args.label_sidecar is not None else None
    )
    args.reuse_features_dir = (
        Path(args.reuse_features_dir).expanduser().resolve()
        if args.reuse_features_dir is not None else None
    )
    args._label_sidecar_payload = _load_label_sidecar(args.label_sidecar)
    args._label_sidecar_sha256 = (
        _sha256(args.label_sidecar) if args.label_sidecar is not None else None
    )
    model = None
    manifests = {}
    for split in args.splits:
        manifest, model = collect_split(args, split=split, model=model)
        manifests[split] = manifest
    summary = {
        "method": "Qwen sample-level uncertainty head feature collection",
        "views": list(args.views),
        "targets": list(TARGETS),
        "splits": {
            split: {
                "num_samples": value["num_samples"],
                "feature_shape": value["feature_shape"],
            }
            for split, value in manifests.items()
        },
        "no_answer_generation_for_primary_view": True,
        "greedy_answer_reused_from_wandb": "greedy_answer" in args.views,
    }
    _atomic_json(Path(args.output_dir) / "collection_summary.json", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    main(build_parser().parse_args())
