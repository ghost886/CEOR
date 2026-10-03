"""OSD auxiliary generation and teacher construction (manuscript Eqs. 1--9).

The original implementation uses the historical names LGD / lgd_uq; its
formulas and adapters are retained in uncertainty/, including resume support.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from ceor.io import atomic_json, image_digest, read_jsonl
from uncertainty.models.lgd_reference_models import (
    load_reference_ensemble, usable_reference_samples,
)
from uncertainty.uncertainty_measures.lgd_uq import LGDUQConfig, recover_lgd_uncertainty

# Public manuscript terminology; do not fork the numerical implementation.
OSDConfig = LGDUQConfig
recover_osd = recover_lgd_uncertainty


def generate_osd(args):
    """Generate auxiliary draws, or recover OSD from already supplied draws.

    This CLI accepts fixed target draws. For joint target+auxiliary generation
    and training-quality filtering use generate_answers.py and prepare-labels.
    """
    input_path, output = Path(args.input), Path(args.output)
    if input_path.resolve() == output.resolve():
        raise ValueError("OSD output must differ from the input JSONL.")
    if args.num_samples < 1 or args.min_models < 1 or args.temperature < 0:
        raise ValueError("Sample/model counts must be positive; temperature non-negative.")
    config = OSDConfig(iou_threshold=args.iou_threshold, smoothing=args.smoothing)
    config.validate()
    if args.reference_config and not args.target_model:
        raise ValueError("--target-model is required to check auxiliary/target independence.")
    ensemble = (load_reference_ensemble(args.reference_config,
                                      target_model_name=args.target_model)
                if args.reference_config else None)
    raw_config = (json.loads(Path(args.reference_config).read_text()) if ensemble else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in read_jsonl(input_path):
                if not row.get("target_predictions"):
                    raise ValueError(f"{row['sample_id']}: target_predictions are required.")
                image = None
                image_size = row.get("image_size")
                image_path = row.get("image_path")
                if image_path:
                    image_path = Path(image_path)
                    if not image_path.is_absolute():
                        image_path = input_path.parent / image_path
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    image_size = image.size
                if ensemble:
                    if image is None:
                        raise ValueError("Live auxiliary generation requires image_path.")
                    from experiments.collect_qwen_uncertainty_heads import grounding_prompt

                    prompt = row.get("prompt") or grounding_prompt(row["question"])
                    identity = {
                        "sample_id": row["sample_id"], "prompt": prompt,
                        "image_sha256": image_digest(image), "config": raw_config,
                        "num_samples": args.num_samples, "temperature": args.temperature,
                        "seed": args.seed,
                    }
                    fingerprint = hashlib.sha256(json.dumps(
                        identity, sort_keys=True, ensure_ascii=False
                    ).encode("utf-8")).hexdigest()
                    cache = Path(args.cache_dir) / fingerprint
                    atomic_json(cache / "identity.json", identity)
                    raw_samples = {}
                    for entry in ensemble.entries:
                        name_hash = hashlib.sha256(entry.name.encode()).hexdigest()
                        path = cache / f"{name_hash}.json"
                        existing = json.loads(path.read_text()) if path.exists() else []
                        raw_samples[entry.name] = ensemble.sample_entry(
                            entry, prompt=prompt, image=image, num_samples=args.num_samples,
                            temperature=args.temperature, base_seed=args.seed,
                            sample_id=row["sample_id"], existing_records=existing,
                            on_record=lambda draws, path=path: atomic_json(path, draws),
                        )
                    auxiliary = usable_reference_samples(raw_samples, allow_empty_models=True)
                    weights = {name: ensemble.weights[name] for name in auxiliary}
                else:
                    auxiliary = row["auxiliary_predictions"]
                    raw_samples = auxiliary
                    weights = row.get("auxiliary_weights")
                if len(auxiliary) < args.min_models:
                    raise ValueError(f"{row['sample_id']}: fewer than {args.min_models} usable auxiliaries.")
                result = recover_osd(
                    ground_truth_box=row["ground_truth_box"],
                    ground_truth_box_format=row.get("ground_truth_box_format", "xyxy"),
                    auxiliary_predictions=auxiliary, target_predictions=row["target_predictions"],
                    auxiliary_weights=weights, image_size=image_size, config=config,
                )
                payload = {
                    "sample_id": row["sample_id"], "question": row.get("question"),
                    "labels": result["uncertainty"], "lgd_uq": result,
                    "auxiliary_samples": raw_samples,
                }
                handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                count += 1
        if not count:
            raise ValueError("Input JSONL is empty.")
        temporary.replace(output)
    finally:
        if ensemble:
            ensemble.close()
    print(f"Saved OSD for {count} samples to {output}")
