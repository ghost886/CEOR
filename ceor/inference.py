"""Frozen-answer CEOR inference using the extracted Qwen3-VL prefill hook."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ceor.io import image_digest, read_jsonl
from ceor.occlusion import mask_image, occlusion_response, parse_prediction_box, layer_ids_for_shape


def frozen_records(*, input_path=None, run_dir=None, split="validation"):
    if input_path:
        input_path = Path(input_path)
        for row in read_jsonl(input_path):
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = input_path.parent / image_path
            yield {**row, "image": str(image_path)}
        return
    root = Path(run_dir)
    root = root / "files" if (root / "files").is_dir() else root
    from experiments.prepare_lgd_quality_filtered_labels import _load_pickle

    records = _load_pickle(root / f"{split}_generations.pkl")
    for sample_id, row in records.items():
        answer = row["most_likely_answer"]
        metadata = row.get("sample_metadata", {})
        yield {
            "sample_id": str(sample_id), "question": row["question"],
            "image": row.get("image", metadata.get("image_path")),
            "frozen_answer": str(answer["response"]),
            "prompt": answer.get("prompt"),
            "group_id": str(metadata.get("image_id", sample_id)),
        }


def score_frozen_prediction(*, answer, image, prompt, key_layer, capture,
                            model_layer_ids=None, clean_feature=None,
                            mask_method="image_mean"):
    """capture(image, prompt) returns [layers,heads,dim]; never calls generate.

    With a matching cached clean feature this performs exactly one diagnostic
    prefill. Without that cache it also recomputes the clean prompt prefill.
    """
    box = parse_prediction_box(answer)
    if box is None:
        return {
            "original_answer": answer, "prediction_box": None, "ceor": None,
            "response_available": False,
            "missing_reason": "unparseable_or_degenerate_prediction", "prefill_passes": 0,
        }
    if model_layer_ids is not None and key_layer == model_layer_ids[-1]:
        return {
            "original_answer": answer, "prediction_box": box, "ceor": None,
            "response_available": False,
            "missing_reason": "no_layers_after_key1", "prefill_passes": 0,
        }
    clean = np.asarray(clean_feature) if clean_feature is not None else capture(image, prompt)
    masked, mask = mask_image(image, box, method=mask_method)
    perturbed = capture(masked, prompt)
    result = occlusion_response(clean, perturbed, key_layer=key_layer,
                                model_layer_ids=model_layer_ids)
    # Only image-mean is the primary CEOR. Other styles are ablations.
    return {
        **result, "original_answer": answer, "prediction_box": box,
        "mask": mask, "primary_method": mask_method == "image_mean",
        "prefill_passes": 1 if clean_feature is not None else 2,
    }


def score(args):
    import torch
    from experiments.collect_qwen_uncertainty_heads import (
        _as_image, collect_answer_free_head_feature, grounding_prompt,
    )
    from uncertainty.models.qwen_vl_models import QwenVLModel

    selection = json.loads(Path(args.key_layer).read_text())
    shape = selection["feature_shape"]
    ids = layer_ids_for_shape(shape, selection["model_layer_ids"])
    if ids != list(range(shape[0])):
        raise ValueError("The Qwen3-VL CLI requires contiguous decoder layers. "
                         "For other architectures supply aligned states to occlusion_response().")
    key_layer = selection["key_layer"]
    if key_layer not in ids:
        raise ValueError("Key layer is not in model_layer_ids.")
    if selection.get("model") and selection["model"] != args.model:
        raise ValueError("The model must match the model used to select Key1.")
    limits = {key: os.getenv(key) for key in ("QWEN_VL_MIN_PIXELS", "QWEN_VL_MAX_PIXELS")}
    if selection.get("processor_limits") and selection["processor_limits"] != limits:
        raise ValueError("Image processor limits differ from the probe collection.")
    cached = {}
    cache_root = None
    if args.features_dir:
        cache_root = Path(args.features_dir) / args.split
        manifest = json.loads((cache_root / "manifest.json").read_text())
        identity = manifest["collection_identity"]
        if identity["model"] != args.model or identity.get("processor_limits") != limits:
            raise ValueError("Cached feature model/processor does not match this run.")
        cached = {str(row["sample_id"]): row for row in manifest["records"]}
        if len(cached) != len(manifest["records"]):
            raise ValueError("Duplicate sample IDs in the clean-feature cache.")
    model = None

    def capture(image, prompt):
        nonlocal model
        if model is None:
            model = QwenVLModel(args.model, max_new_tokens=1)
        feature = collect_answer_free_head_feature(model, prompt=prompt, image=image)["feature"]
        value = feature.float().numpy()
        if list(value.shape) != shape:
            raise ValueError("Captured shape differs from the probe-selected model shape.")
        return value

    output = Path(args.output)
    if args.input and Path(args.input).resolve() == output.resolve():
        raise ValueError("Score output must differ from the input JSONL.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = available = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in frozen_records(input_path=args.input, run_dir=args.run_dir, split=args.split):
            if args.limit and count >= args.limit:
                break
            image = _as_image(row["image"])
            prompt = row.get("prompt") or grounding_prompt(row["question"])
            answer = row["frozen_answer"]
            clean = None
            if cache_root is not None:
                source = cached.get(row["sample_id"])
                if source is None or source["prompt"] != prompt or source["greedy_answer"] != answer:
                    raise ValueError(f"{row['sample_id']}: cached prompt/answer/sample does not match.")
                path = cache_root / source["feature_files"]["answer_free"]
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if str(saved["sample_id"]) != row["sample_id"]:
                    raise ValueError("Cached feature sample ID does not match.")
                if saved.get("image_sha256") != image_digest(image):
                    raise ValueError("Cached image digest missing/different; recollect or omit --features-dir.")
                clean = torch.as_tensor(saved["feature"]).float().numpy()
                if list(clean.shape) != shape:
                    raise ValueError("Cached clean feature shape does not match the probe.")
            result = score_frozen_prediction(
                answer=answer, image=image, prompt=prompt, key_layer=key_layer,
                capture=capture, model_layer_ids=ids, clean_feature=clean,
                mask_method=args.mask_method,
            )
            if not args.save_head_responses:
                result.pop("relative_l2_per_head", None)
                result.pop("cosine_per_head", None)
            payload = {"sample_id": row["sample_id"], "group_id": row.get("group_id"), **result}
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            count += 1
            available += int(result["response_available"])
    if not count:
        raise ValueError("No frozen predictions were supplied.")
    temporary.replace(output)
    print(f"Saved {count} frozen predictions to {output}; CEOR available for {available}/{count}.")
