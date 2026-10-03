"""Sample answers from LLMs on QA task."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import os
# Generation artifacts can be several gigabytes, so keep W&B local by default.
# Set WANDB_MODE=online explicitly when remote synchronization is desired.
os.environ.setdefault("WANDB_MODE", "offline")
import logging
import pickle
import random
from collections import defaultdict
import threading
from typing import Any, Mapping

from tqdm import tqdm

import numpy as np
import torch
try:
    import wandb
except ImportError:  # pragma: no cover - local-only fallback
    from uncertainty.utils import wandb_stub as wandb
from PIL import Image

from analyze_results import (
    filter_reportable_uncertainty_measures,
    is_retired_uncertainty_measure,
)
from uncertainty.data.data_utils import GROUNDING_DATASETS, VLM_DATASETS, load_ds
from uncertainty.utils import utils
from uncertainty.utils.generation_checkpoints import (
    GenerationCheckpointStore,
    build_lgd_reference_sidecar,
    generation_fingerprint,
)
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.uncertainty_measures.cross_model_semantic_uq import (
    CrossModelSemanticUQConfig,
    SentenceT5SimilarityEncoder,
    compute_cross_model_semantic_uq,
)
from uncertainty.models.lgd_reference_models import (
    load_reference_ensemble,
    usable_reference_samples,
)
from uncertainty.uncertainty_measures.lgd_uq import (
    LGDUQConfig,
    recover_lgd_iou_ablation,
)

import warnings
warnings.filterwarnings("ignore")

utils.setup_logger()

LGD_IOU_ABLATION_THRESHOLDS = (0.3, 0.7, 0.9)
_ACTIVE_LGD_PREFETCH_HANDLE = None


def _cross_model_semantic_uq_enabled(args) -> bool:
    """Enable the paper baseline automatically whenever LGD sampling is active."""
    if not getattr(args, "collect_lgd_uq", False):
        return False
    # ``None`` means that the user did not make an explicit choice.  LGD already
    # collects every response required by the paper baseline, so compute it by
    # default.  ``--no-collect_cross_model_semantic_uq`` remains an opt-out for
    # machines on which sentence-T5-xl is unavailable.
    return getattr(args, "collect_cross_model_semantic_uq", None) is not False


def _configure_primary_trajectories(args):
    """Resolve the Qwen default before checkpoint identity/model construction."""
    requested = getattr(args, "collect_primary_trajectories", None)
    supported = "qwen" in str(getattr(args, "model_name", "")).lower()
    enabled = supported if requested is None else bool(requested)
    if enabled and not supported:
        raise ValueError("--collect_primary_trajectories currently requires the Qwen wrapper.")
    args.collect_primary_trajectories = enabled
    if enabled:
        # Prevent resuming older shards that did not contain SIVR features.
        args.primary_trajectory_schema_version = 2
    if enabled:
        args.collect_probe_features = True
        args.collect_icr_probe = True
        args.icr_save_direction_vectors = True
        logging.info(
            "Primary-answer trajectories enabled: exact token IDs, ICR Attn/Proj, "
            "answer-free heads, CoE and layer entropy/convergence."
        )


def _configure_dataset_task(args):
    """Apply the task contract required by registered grounding datasets."""
    if args.dataset not in GROUNDING_DATASETS:
        return
    if args.metric != "grounding":
        logging.info(
            "Forcing --metric=grounding for visual-grounding dataset `%s`.",
            args.dataset,
        )
        args.metric = "grounding"
    if args.prompt_type != "grounding":
        logging.info(
            "Forcing --prompt_type=grounding for visual-grounding dataset `%s`.",
            args.dataset,
        )
        args.prompt_type = "grounding"


def _configure_lgd_uq(args):
    """Validate the optional LGD teacher without changing default generation."""
    if (
        getattr(args, "collect_cross_model_semantic_uq", None) is True
        and not getattr(args, "collect_lgd_uq", False)
    ):
        raise ValueError(
            "--collect_cross_model_semantic_uq requires --collect_lgd_uq so that "
            "target and auxiliary response samples are available."
        )
    if not getattr(args, "collect_lgd_uq", False):
        return
    if args.metric != "grounding":
        raise ValueError("--collect_lgd_uq currently supports only visual grounding.")
    if not args.lgd_reference_config:
        raise ValueError("--collect_lgd_uq requires --lgd_reference_config.")
    if not os.path.isfile(args.lgd_reference_config):
        raise FileNotFoundError(
            f"LGD reference config does not exist: {args.lgd_reference_config}"
        )
    if int(args.lgd_num_reference_samples) < 1:
        raise ValueError("--lgd_num_reference_samples must be positive.")
    if int(args.lgd_num_target_samples) < 1:
        raise ValueError("--lgd_num_target_samples must be positive.")
    if int(getattr(args, "lgd_requests_per_model", 1)) < 1:
        raise ValueError("--lgd_requests_per_model must be positive.")
    if float(args.lgd_reference_temperature) <= 0.0:
        raise ValueError("--lgd_reference_temperature must be positive for sampling.")
    if float(args.temperature) <= 0.0:
        raise ValueError("--temperature must be positive for LGD target sampling.")
    for threshold in {
        float(args.lgd_iou_threshold),
        *(
            float(value)
            for value in getattr(
                args,
                "lgd_iou_ablation_thresholds",
                LGD_IOU_ABLATION_THRESHOLDS,
            )
        ),
    }:
        LGDUQConfig(
            iou_threshold=threshold,
            smoothing=float(args.lgd_smoothing),
            qwen_coordinate_scale=float(args.lgd_qwen_coordinate_scale),
        ).validate()


def _lgd_image_size(example, image):
    width, height = example.get("image_width"), example.get("image_height")
    if width is not None and height is not None and int(width) > 0 and int(height) > 0:
        return int(width), int(height)
    if isinstance(image, Image.Image):
        return int(image.width), int(image.height)
    return None


def _sequence_log_probability(token_log_likelihoods):
    if token_log_likelihoods is None:
        return None
    if torch.is_tensor(token_log_likelihoods):
        values = token_log_likelihoods.detach().float().cpu().reshape(-1).tolist()
    else:
        try:
            values = np.asarray(token_log_likelihoods, dtype=float).reshape(-1).tolist()
        except (TypeError, ValueError):
            return None
    if not values or not np.isfinite(values).all():
        return None
    return float(sum(values))


def _collect_lgd_uq(
    *,
    args,
    reference_ensemble,
    example,
    image,
    prompt,
    target_responses,
    semantic_uq_encoder=None,
    reference_samples=None,
):
    """Collect reference samples and create one LGD uncertainty teacher label."""
    if image is None:
        raise ValueError("LGD reference sampling requires an image.")
    sample_id = example["question_id"]
    if reference_samples is None:
        reference_samples = reference_ensemble.sample(
            prompt=prompt,
            image=image,
            num_samples=int(args.lgd_num_reference_samples),
            temperature=float(args.lgd_reference_temperature),
            base_seed=int(args.random_seed),
            sample_id=sample_id,
        )
    reference_predictions = usable_reference_samples(
        reference_samples,
        allow_empty_models=True,
    )
    if not reference_predictions:
        raise RuntimeError(
            f"All LGD reference models rejected sample {sample_id!r}; no "
            "reference distribution can be estimated for this example."
        )
    unavailable_reference_models = sorted(
        set(reference_samples) - set(reference_predictions)
    )
    for model_name, raw_records in reference_samples.items():
        included_count = len(reference_predictions.get(model_name, ()))
        excluded_count = len(raw_records) - included_count
        if excluded_count:
            logging.warning(
                "LGD-UQ reference %r: using %d/%d draws; %d provider-filtered "
                "draw(s) are retained for audit but excluded from uncertainty.",
                model_name,
                included_count,
                len(raw_records),
                excluded_count,
            )
    target_records = []
    for response, token_log_likelihoods, _, _ in target_responses[
        : int(args.lgd_num_target_samples)
    ]:
        target_records.append({
            "response": response,
            "token_log_likelihoods": token_log_likelihoods,
            "sequence_log_probability": _sequence_log_probability(
                token_log_likelihoods
            ),
        })
    if len(target_records) != int(args.lgd_num_target_samples):
        raise RuntimeError(
            "LGD target sampling did not produce the requested number of responses."
        )
    if "bbox" in example:
        ground_truth_box = example["bbox"]
        ground_truth_format = example.get("bbox_format", "xyxy")
    elif "answer" in example:
        ground_truth_box = example["answer"]
        ground_truth_format = "xyxy"
    else:
        raise ValueError("LGD requires a hard ground-truth box in bbox or answer.")
    primary_threshold = float(args.lgd_iou_threshold)
    thresholds = [
        primary_threshold,
        *(
            float(value)
            for value in getattr(
                args,
                "lgd_iou_ablation_thresholds",
                LGD_IOU_ABLATION_THRESHOLDS,
            )
        ),
    ]
    ablation_results = recover_lgd_iou_ablation(
        ground_truth_box=ground_truth_box,
        ground_truth_box_format=ground_truth_format,
        auxiliary_predictions=reference_predictions,
        iou_thresholds=thresholds,
        target_predictions=target_records,
        auxiliary_weights={
            name: reference_ensemble.weights[name]
            for name in reference_predictions
        },
        image_size=_lgd_image_size(example, image),
        config=LGDUQConfig(
            iou_threshold=primary_threshold,
            smoothing=float(args.lgd_smoothing),
            qwen_coordinate_scale=float(args.lgd_qwen_coordinate_scale),
        ),
    )
    result = ablation_results[f"{primary_threshold:g}"]
    # Preserve enough information to audit how the modes and assignments change
    # without duplicating the expensive raw API sampling records.
    result["iou_threshold_ablation"] = {
        "primary_threshold": primary_threshold,
        "thresholds": [float(key) for key in ablation_results],
        "results": {
            key: {
                "config": candidate["config"],
                "ideal_answer_distribution": candidate["ideal_answer_distribution"],
                "auxiliary_models": candidate["auxiliary_models"],
                "target_model": candidate["target_model"],
                "uncertainty": candidate["uncertainty"],
                "diagnostics": candidate["diagnostics"],
            }
            for key, candidate in ablation_results.items()
        },
    }
    # Preserve decoded tokens and sequence likelihoods for auditing/recovery;
    # the pure estimator above stores parsed boxes, assignments, and counts.
    result["sampling"] = {
        "auxiliary": reference_samples,
        "auxiliary_requested_per_model": int(args.lgd_num_reference_samples),
        "auxiliary_used_per_model": {
            name: len(reference_predictions.get(name, ()))
            for name in reference_samples
        },
        "auxiliary_excluded_per_model": {
            name: len(reference_samples[name])
            - len(reference_predictions.get(name, ()))
            for name in reference_samples
        },
        "auxiliary_unavailable_models": unavailable_reference_models,
        "target": target_records,
        "fixed_input": True,
    }
    if semantic_uq_encoder is not None:
        result["cross_model_semantic_uq"] = compute_cross_model_semantic_uq(
            target_responses=target_records,
            auxiliary_responses=reference_predictions,
            encoder=semantic_uq_encoder,
            # Equation (3) in the paper averages the m auxiliary models
            # uniformly.  LGD model weights belong only to the IoU estimator.
            auxiliary_weights=None,
        )
    return result


def _example_image(example, *, open_path: bool):
    """Return the image field using the same precedence throughout the pipeline."""
    if "image" in example:
        image = example.get("image")
    elif "image_1" in example:
        image = example.get("image_1")
    else:
        image = None
    if open_path and isinstance(image, str) and os.path.exists(image):
        return Image.open(image).convert("RGB")
    return image


def _select_generation_splits(
    args,
    *,
    train_dataset,
    validation_dataset,
    remaining_answerable,
    unanswerable_indices,
):
    """Freeze both split selections before any expensive generation begins."""
    plan = {}
    for dataset_split in ("train", "validation"):
        if dataset_split == "train":
            if not args.get_training_set_generations:
                continue
            dataset = train_dataset
            # Some evaluation-only datasets (currently PR-Bench) intentionally
            # expose no training records. Keep the empty train split out of
            # the generation plan so downstream aggregation never sees an
            # empty accuracy list.
            if len(dataset) == 0:
                logging.info(
                    "Skipping train generation because the loaded dataset has "
                    "no training records."
                )
                continue
            possible_indices = sorted(
                set(remaining_answerable) | set(unanswerable_indices)
            )
        else:
            dataset = validation_dataset
            possible_indices = list(range(len(dataset)))

        split_num_samples = getattr(args, f"{dataset_split}_num_samples", None)
        if split_num_samples is None:
            split_num_samples = args.num_samples
        if split_num_samples <= 0:
            raise ValueError(
                f"--{dataset_split}_num_samples/--num_samples must be positive, "
                f"got {split_num_samples}."
            )
        if len(possible_indices) <= split_num_samples:
            indices = possible_indices
        else:
            split_rng = random.Random(
                f"{int(args.random_seed)}|generation-split|{dataset_split}"
            )
            indices = split_rng.sample(possible_indices, split_num_samples)
        if split_num_samples > len(possible_indices):
            logging.warning(
                "Requested %d %s samples, but only %d are available. Using all "
                "available samples.",
                split_num_samples,
                dataset_split,
                len(possible_indices),
            )
        logging.info(
            "Selected %d of %d available %s samples.",
            len(indices),
            len(possible_indices),
            dataset_split,
        )
        plan[dataset_split] = {"dataset": dataset, "indices": indices}
    return plan


def _sample_indices(indices, requested, *, label):
    """Sample up to ``requested`` indices without failing on short splits."""
    requested = int(requested)
    if requested < 0:
        raise ValueError(f"{label} must be non-negative, got {requested}.")
    available = list(indices)
    count = min(requested, len(available))
    if requested > len(available):
        logging.warning(
            "Requested %d %s examples, but only %d are available; using all "
            "available examples.",
            requested,
            label,
            len(available),
        )
    return random.sample(available, count)


def _reference_config_checkpoint_identity(config):
    """Exclude transport-only tuning so retry changes keep valid model caches."""
    normalized = json.loads(json.dumps(config))
    for key in (
        "infrastructure_max_retries",
        "infrastructure_retry_delay",
        "infrastructure_retry_max_delay",
    ):
        normalized.pop(key, None)
    for model_spec in normalized.get("models", []):
        if isinstance(model_spec, dict):
            model_spec.pop("timeout", None)
            model_spec.pop("max_retries", None)
    return normalized


def _make_lgd_reference_jobs(args, split_plan, make_prompt, brief):
    jobs = []
    enabled_splits = set(args.lgd_splits)
    for dataset_split, split_info in split_plan.items():
        if dataset_split not in enabled_splits:
            continue
        dataset = split_info["dataset"]
        for index in split_info["indices"]:
            example = dataset[index]
            image = _example_image(example, open_path=False)
            if image is None:
                raise ValueError(
                    f"LGD reference sampling requires an image for sample "
                    f"{example['question_id']!r}."
                )
            image_filename = getattr(image, "filename", None)
            if image_filename and os.path.exists(image_filename):
                # Hugging Face datasets often decode paths into PIL objects.
                # Retaining the path keeps the prefetch job table lightweight;
                # each worker opens its own image safely.
                image = image_filename
            jobs.append({
                "split": dataset_split,
                "sample_id": example["question_id"],
                "prompt": make_prompt(
                    None,
                    example["question"],
                    None,
                    brief,
                    args.brief_always and args.enable_brief,
                ),
                "image": image,
            })
    return jobs


def _prefetch_lgd_references(
    *,
    args,
    reference_ensemble,
    jobs,
    checkpoint_store,
    stop_event=None,
):
    """Run resumable model/sample producers without coupling their cache shards."""
    if not jobs:
        return
    requested_draws = int(args.lgd_num_reference_samples)
    configured_workers = int(args.lgd_parallel_workers)
    worker_count = (
        len(reference_ensemble.entries)
        if configured_workers <= 0
        else min(configured_workers, len(reference_ensemble.entries))
    )
    worker_count = max(1, worker_count)
    stop_event = stop_event or threading.Event()

    remaining_draws_by_model = {}
    complete_samples_by_model = {}
    for entry in reference_ensemble.entries:
        remaining_draws = 0
        complete_samples = 0
        for job in jobs:
            cached = checkpoint_store.load_reference(
                job["split"], entry.name, job["sample_id"]
            )
            cached_draws = min(len(cached), requested_draws)
            remaining_draws += requested_draws - cached_draws
            if cached_draws == requested_draws:
                complete_samples += 1
        remaining_draws_by_model[entry.name] = remaining_draws
        complete_samples_by_model[entry.name] = complete_samples

    def run_job_lane(entry, lane_jobs):
        resumed_samples = 0
        generated_draws = 0
        checked_samples = 0
        for _, job in lane_jobs:
            if stop_event.is_set():
                break
            checked_samples += 1
            existing = checkpoint_store.load_reference(
                job["split"], entry.name, job["sample_id"]
            )
            if len(existing) >= requested_draws:
                resumed_samples += 1
                continue
            before = len(existing)
            source_image = job["image"]
            reference_image = (
                source_image.copy()
                if isinstance(source_image, Image.Image)
                else source_image
            )
            try:
                current_records = existing
                # Request one draw at a time.  Besides making every successful
                # POST immediately durable, this lets Ctrl-C stop between draws
                # instead of waiting for the rest of a sample's batch.
                while (
                    len(current_records) < requested_draws
                    and not stop_event.is_set()
                ):
                    current_records = reference_ensemble.sample_entry(
                        entry,
                        prompt=job["prompt"],
                        image=reference_image,
                        num_samples=len(current_records) + 1,
                        temperature=float(args.lgd_reference_temperature),
                        base_seed=int(args.random_seed),
                        sample_id=job["sample_id"],
                        existing_records=current_records,
                        on_record=lambda records, current_job=job: (
                            checkpoint_store.save_reference(
                                current_job["split"],
                                entry.name,
                                current_job["sample_id"],
                                records,
                            )
                        ),
                    )
            finally:
                if reference_image is not source_image:
                    reference_image.close()
            generated_draws += len(current_records) - before
        return {
            "checked_samples": checked_samples,
            "resumed_samples": resumed_samples,
            "generated_draws": generated_draws,
        }

    def run_model(entry):
        requested_lanes = max(
            1, int(getattr(args, "lgd_requests_per_model", 1))
        )
        entry_type = entry.adapter.describe().get("type")
        lane_entries = [entry]
        if entry_type != "huggingface":
            for _ in range(requested_lanes - 1):
                clone = reference_ensemble.clone_entry(entry)
                if clone is None:
                    break
                lane_entries.append(clone)

        lane_count = len(lane_entries)
        indexed_jobs = list(enumerate(jobs, start=1))
        lane_job_lists = [indexed_jobs[lane::lane_count] for lane in range(lane_count)]
        summaries = []
        try:
            if lane_count == 1:
                summaries.append(run_job_lane(lane_entries[0], lane_job_lists[0]))
            else:
                with ThreadPoolExecutor(
                    max_workers=lane_count,
                    thread_name_prefix=f"lgd-{entry.name[:20]}",
                ) as lane_executor:
                    lane_futures = [
                        lane_executor.submit(run_job_lane, lane_entry, lane_jobs)
                        for lane_entry, lane_jobs in zip(
                            lane_entries, lane_job_lists
                        )
                    ]
                    for lane_future in as_completed(lane_futures):
                        try:
                            summaries.append(lane_future.result())
                        except BaseException:
                            # Stop sibling lanes before propagating the failure.
                            # In-flight HTTP calls cannot be cancelled, but every
                            # lane observes this flag before starting another draw.
                            stop_event.set()
                            for sibling in lane_futures:
                                sibling.cancel()
                            raise
        finally:
            for clone in lane_entries[1:]:
                clone.adapter.close()

        summary = {
            "model": entry.name,
            "request_lanes": lane_count,
            "checked_samples": sum(
                item["checked_samples"] for item in summaries
            ),
            "resumed_samples": sum(
                item["resumed_samples"] for item in summaries
            ),
            "generated_draws": sum(
                item["generated_draws"] for item in summaries
            ),
        }
        logging.info("LGD prefetch worker complete: %s", summary)
        return summary

    logging.info(
        "Starting LGD reference prefetch: %d models, %d model workers, up to "
        "%d concurrent requests/model, %d candidate samples/model. Cached "
        "complete samples/model: %s. Remaining draws/model: %s.",
        len(reference_ensemble.entries),
        worker_count,
        max(1, int(getattr(args, "lgd_requests_per_model", 1))),
        len(jobs),
        complete_samples_by_model,
        remaining_draws_by_model,
    )
    executor = ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="lgd-reference"
    )
    futures = {
        executor.submit(run_model, entry): entry.name
        for entry in reference_ensemble.entries
    }
    failures = []
    try:
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                failures.append((model_name, exc))
                logging.exception("LGD prefetch worker %r failed.", model_name)
                # A complete reference ensemble is required for scientifically
                # valid aggregation.  Stop all other producers at the next draw
                # boundary while retaining every shard already written.
                stop_event.set()
                for sibling in futures:
                    sibling.cancel()
                break
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if failures:
        details = "; ".join(
            f"{model_name}: {type(exc).__name__}: {exc}"
            for model_name, exc in failures
        )
        raise RuntimeError(
            "One or more LGD reference workers failed. Completed model/sample "
            "shards were retained. After fixing the provider account/config, "
            "resume this exact run with "
            f"--generation_checkpoint_dir {checkpoint_store.root} "
            f"--resume_generation: {details}"
        ) from failures[0][1]


class _LGDReferencePrefetchHandle:
    """Background coordinator for reference-model producers."""

    def __init__(self, *, args, reference_ensemble, jobs, checkpoint_store):
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lgd-prefetch-coordinator"
        )
        self.future = self.executor.submit(
            _prefetch_lgd_references,
            args=args,
            reference_ensemble=reference_ensemble,
            jobs=jobs,
            checkpoint_store=checkpoint_store,
            stop_event=self.stop_event,
        )
        self._closed = False
        self.future.add_done_callback(self._log_early_failure)

    @staticmethod
    def _log_early_failure(future):
        if future.cancelled():
            return
        exception = future.exception()
        if exception is not None:
            logging.error(
                "Background LGD reference production failed; target generation "
                "will continue and all completed shards remain resumable: %s",
                exception,
            )

    def wait(self):
        try:
            return self.future.result()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True

    def stop(self):
        if self._closed:
            return
        self.stop_event.set()
        self.future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True


def _load_cached_lgd_references(
    *,
    args,
    reference_ensemble,
    checkpoint_store,
    dataset_split,
    sample_id,
    require_complete=True,
):
    requested_draws = int(args.lgd_num_reference_samples)
    records = {}
    for entry in reference_ensemble.entries:
        model_records = checkpoint_store.load_reference(
            dataset_split, entry.name, sample_id
        )
        if len(model_records) != requested_draws:
            if not require_complete:
                return None
            raise RuntimeError(
                f"LGD cache for {entry.name!r}, split {dataset_split!r}, sample "
                f"{sample_id!r} has {len(model_records)}/{requested_draws} draws."
            )
        records[entry.name] = model_records
    return records


def _lgd_cache_matches_reference_ensemble(lgd_result, reference_ensemble):
    """Reject derived LGD data inherited from a different reference set.

    Target-generation shards can be shared when a reference model is added or
    removed. The raw target responses remain valid in that case, but an
    embedded ``lgd_uq`` result must be rebuilt from the current reference
    shards instead of being reused.
    """
    if not isinstance(lgd_result, Mapping):
        return False
    sampling = lgd_result.get("sampling")
    auxiliary = sampling.get("auxiliary") if isinstance(sampling, Mapping) else None
    if not isinstance(auxiliary, Mapping):
        return False
    expected = {entry.name for entry in reference_ensemble.entries}
    return set(auxiliary) == expected


def _can_reference_only_resume(args, checkpoint_store, split_sample_ids):
    """Return true only when no target-model or p_true call remains."""
    if not (
        bool(getattr(args, "resume_generation", False))
        and bool(getattr(args, "collect_lgd_uq", False))
    ):
        return False
    target_cache_complete = all(
        checkpoint_store.load_generation(dataset_split, sample_id) is not None
        for dataset_split, sample_ids in split_sample_ids.items()
        if dataset_split != "p_true_few_shot"
        for sample_id in sample_ids
    )
    p_true_cache_complete = (
        not bool(getattr(args, "compute_p_true", False))
        or isinstance(checkpoint_store.load_state("p_true_setup"), Mapping)
    )
    return target_cache_complete and p_true_cache_complete


def _sample_metadata(example, image) -> dict:
    """Keep only scalar/list metadata needed by grouped RQ1/RQ2 analysis."""
    keys = (
        "dataset_name", "split", "source_id", "image_id", "ref_id", "ann_id",
        "image_width", "image_height", "bbox", "bbox_format",
        "distractor_bbox", "distractor_bbox_number",
        "source_split", "target_policy", "phrase_id", "command_token",
        "scene_token", "sample_token", "box_token", "object_name",
        "category_name",
    )
    metadata = {}
    for key in keys:
        value = example.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, np.ndarray):
            value = value.reshape(-1).tolist()
        if isinstance(value, (str, int, float, bool, list, tuple)):
            metadata[key] = list(value) if isinstance(value, tuple) else value
    image_path = None
    if isinstance(image, str):
        image_path = image
    elif hasattr(image, "filename") and getattr(image, "filename", None):
        image_path = str(image.filename)
    if image_path:
        metadata["image_path"] = image_path
    return metadata


def _finalize_lgd_pipeline(
    *,
    args,
    split_plan,
    checkpoint_store,
    reference_ensemble,
    semantic_uq_encoder,
    make_prompt,
    brief,
    experiment_details,
):
    """Join target/reference shards and materialize LGD/semantic outputs."""
    enabled_splits = set(args.lgd_splits)
    for dataset_split, split_info in split_plan.items():
        if dataset_split not in enabled_splits:
            continue

        dataset = split_info["dataset"]
        indices = split_info["indices"]
        generations = {}
        lgd_measures = defaultdict(list)
        lgd_iou_measures = defaultdict(lambda: defaultdict(list))
        semantic_measures = defaultdict(list)
        newly_finalized = 0
        references_by_sample = {}

        logging.info(
            "Finalizing LGD/semantic UQ for %d %s target/reference shard pairs.",
            len(indices),
            dataset_split,
        )
        for index in tqdm(indices, desc=f"finalize-lgd-{dataset_split}"):
            example = dataset[index]
            sample_id = example["question_id"]
            record = checkpoint_store.load_generation(dataset_split, sample_id)
            if record is None:
                raise RuntimeError(
                    f"Target generation cache is missing for split "
                    f"{dataset_split!r}, sample {sample_id!r}. Re-run with the "
                    "same checkpoint directory to resume target generation."
                )

            lgd_result = record.get("lgd_uq")
            record_changed = False
            reference_samples = _load_cached_lgd_references(
                args=args,
                reference_ensemble=reference_ensemble,
                checkpoint_store=checkpoint_store,
                dataset_split=dataset_split,
                sample_id=sample_id,
                require_complete=True,
            )
            references_by_sample[sample_id] = reference_samples
            if isinstance(lgd_result, Mapping) and not _lgd_cache_matches_reference_ensemble(
                lgd_result, reference_ensemble
            ):
                cached_sampling = lgd_result.get("sampling")
                cached_auxiliary = (
                    cached_sampling.get("auxiliary")
                    if isinstance(cached_sampling, Mapping)
                    else None
                )
                cached_names = (
                    sorted(cached_auxiliary)
                    if isinstance(cached_auxiliary, Mapping)
                    else []
                )
                current_names = sorted(entry.name for entry in reference_ensemble.entries)
                logging.info(
                    "Recomputing stale LGD cache for %s/%s: cached reference "
                    "models=%s, current reference models=%s.",
                    dataset_split,
                    sample_id,
                    cached_names,
                    current_names,
                )
                lgd_result = None
            if not isinstance(lgd_result, Mapping):
                source_image = _example_image(example, open_path=False)
                image = _example_image(example, open_path=True)
                if image is None:
                    raise ValueError(
                        f"LGD finalization requires an image for sample "
                        f"{sample_id!r}."
                    )
                current_input = make_prompt(
                    None,
                    example["question"],
                    None,
                    brief,
                    args.brief_always and args.enable_brief,
                )
                try:
                    lgd_result = _collect_lgd_uq(
                        args=args,
                        reference_ensemble=reference_ensemble,
                        example=example,
                        image=image,
                        prompt=current_input,
                        target_responses=record.get("responses", []),
                        semantic_uq_encoder=semantic_uq_encoder,
                        reference_samples=reference_samples,
                    )
                finally:
                    if image is not source_image and isinstance(image, Image.Image):
                        image.close()
                lgd_result["sampling"]["auxiliary_cache"] = {
                    "root": str(checkpoint_store.root.resolve()),
                    "split": dataset_split,
                    "layout": "lgd_references/<split>/<model>/<sample>.pkl",
                    "per_model_independent": True,
                }
                record["lgd_uq"] = lgd_result
                newly_finalized += 1
                record_changed = True

            portable_sidecar = f"{dataset_split}_lgd_references.pkl"
            auxiliary_cache = lgd_result.setdefault("sampling", {}).setdefault(
                "auxiliary_cache", {}
            )
            if auxiliary_cache.get("portable_sidecar") != portable_sidecar:
                auxiliary_cache["portable_sidecar"] = portable_sidecar
                auxiliary_cache["sample_key"] = sample_id
                record_changed = True

            uncertainty = lgd_result["uncertainty"]
            for metric in ("u_a", "u_d", "u_e", "u_total"):
                lgd_measures[metric].append(float(uncertainty[metric]))
            for threshold_key, threshold_result in (
                lgd_result.get("iou_threshold_ablation", {})
                .get("results", {})
                .items()
            ):
                for metric in ("u_a", "u_d", "u_e", "u_total"):
                    lgd_iou_measures[threshold_key][metric].append(
                        float(threshold_result["uncertainty"][metric])
                    )

            semantic_result = lgd_result.get("cross_model_semantic_uq")
            if isinstance(semantic_result, Mapping):
                if record.get("cross_model_semantic_uq") is not semantic_result:
                    record["cross_model_semantic_uq"] = semantic_result
                    record_changed = True
                semantic_uncertainty = semantic_result["uncertainty"]
                for metric in ("u_aleatoric", "u_epistemic", "u_total"):
                    semantic_measures[metric].append(
                        float(semantic_uncertainty[metric])
                    )

            if record_changed:
                checkpoint_store.save_generation(
                    dataset_split, sample_id, record
                )
            generations[sample_id] = record

        utils.save(generations, f"{dataset_split}_generations.pkl")
        sidecar_filename = f"{dataset_split}_lgd_references.pkl"
        sidecar = build_lgd_reference_sidecar(
            split=dataset_split,
            sample_ids=list(generations),
            model_names=[entry.name for entry in reference_ensemble.entries],
            num_draws_per_model=int(args.lgd_num_reference_samples),
            references_by_sample=references_by_sample,
            checkpoint_root=str(checkpoint_store.root.resolve()),
        )
        utils.save(sidecar, sidecar_filename)
        experiment_details.setdefault("lgd_reference_sidecars", {})[
            dataset_split
        ] = {
            "filename": sidecar_filename,
            "num_samples": len(generations),
            "model_names": sidecar["model_names"],
            "num_draws_per_model": sidecar["num_draws_per_model"],
            "sample_order_matches_generations": True,
        }
        logging.info(
            "Saved portable LGD reference sidecar beside generations: %s",
            sidecar_filename,
        )
        checkpoint_store.mark_split_complete(
            dataset_split, num_samples=len(generations)
        )

        print(
            f"LGD-UQ {dataset_split} final mean: "
            f"U_A={np.mean(lgd_measures['u_a']):.6f} "
            f"U_D={np.mean(lgd_measures['u_d']):.6f} "
            f"U_E={np.mean(lgd_measures['u_e']):.6f} "
            f"U_T={np.mean(lgd_measures['u_total']):.6f}"
        )
        logging.info(
            "LGD finalization complete for %s: %d newly joined, %d resumed.",
            dataset_split,
            newly_finalized,
            len(generations) - newly_finalized,
        )
        for threshold_key in sorted(lgd_iou_measures, key=float):
            values = lgd_iou_measures[threshold_key]
            print(
                f"LGD-UQ {dataset_split} IoU={threshold_key} mean: "
                f"U_A={np.mean(values['u_a']):.6f} "
                f"U_D={np.mean(values['u_d']):.6f} "
                f"U_E={np.mean(values['u_e']):.6f} "
                f"U_T={np.mean(values['u_total']):.6f}"
            )
        wandb.log({
            f"{dataset_split}_lgd_u_a_mean": float(np.mean(lgd_measures["u_a"])),
            f"{dataset_split}_lgd_u_d_mean": float(np.mean(lgd_measures["u_d"])),
            f"{dataset_split}_lgd_u_e_mean": float(np.mean(lgd_measures["u_e"])),
            f"{dataset_split}_lgd_u_total_mean": float(
                np.mean(lgd_measures["u_total"])
            ),
        })

        if semantic_measures:
            semantic_means = {
                metric: float(np.mean(values))
                for metric, values in semantic_measures.items()
            }
            print(
                f"Original-paper cross-model semantic UQ {dataset_split} mean: "
                f"AU={semantic_means['u_aleatoric']:.6f} "
                f"EU={semantic_means['u_epistemic']:.6f} "
                f"TU={semantic_means['u_total']:.6f}"
            )
            experiment_details[dataset_split][
                "cross_model_semantic_uq_summary"
            ] = {
                "num_samples": len(semantic_measures["u_total"]),
                "mean_au": semantic_means["u_aleatoric"],
                "mean_eu": semantic_means["u_epistemic"],
                "mean_tu": semantic_means["u_total"],
            }
            wandb.log({
                f"{dataset_split}_cross_model_semantic_au_mean": semantic_means[
                    "u_aleatoric"
                ],
                f"{dataset_split}_cross_model_semantic_eu_mean": semantic_means[
                    "u_epistemic"
                ],
                f"{dataset_split}_cross_model_semantic_tu_mean": semantic_means[
                    "u_total"
                ],
            })

        if dataset_split == "validation":
            results_path = os.path.join(
                wandb.run.dir, "uncertainty_measures.pkl"
            )
            results_dict = {}
            if os.path.isfile(results_path):
                with open(results_path, "rb") as handle:
                    cached_results = pickle.load(handle)
                if isinstance(cached_results, Mapping):
                    results_dict = dict(cached_results)
            uncertainty_measures = dict(
                results_dict.get("uncertainty_measures", {})
            )
            uncertainty_measures.update({
                "lgd_u_a": lgd_measures["u_a"],
                "lgd_u_d": lgd_measures["u_d"],
                "lgd_u_e": lgd_measures["u_e"],
                "lgd_u_total": lgd_measures["u_total"],
            })
            for threshold_key, threshold_values in lgd_iou_measures.items():
                safe_threshold = threshold_key.replace(".", "_")
                for metric, values in threshold_values.items():
                    uncertainty_measures[
                        f"lgd_iou_{safe_threshold}_{metric}"
                    ] = values
            if semantic_measures:
                uncertainty_measures.update({
                    "cross_model_semantic_au": semantic_measures["u_aleatoric"],
                    "cross_model_semantic_eu": semantic_measures["u_epistemic"],
                    "cross_model_semantic_tu": semantic_measures["u_total"],
                })
            results_dict["uncertainty_measures"] = (
                filter_reportable_uncertainty_measures(uncertainty_measures)
            )
            utils.save(results_dict, "uncertainty_measures.pkl")


def main(args):
    global _ACTIVE_LGD_PREFETCH_HANDLE
    _configure_dataset_task(args)
    _configure_primary_trajectories(args)
    _configure_lgd_uq(args)

    if getattr(args, "collect_icr_probe", False) and not args.collect_probe_features:
        logging.info(
            "Enabling --collect_probe_features because ICR vectors are saved "
            "inside each generation's probe_features."
        )
        args.collect_probe_features = True

    if (
        getattr(args, "collect_vib_probe", False)
        or getattr(args, "collect_lrp_probe", False)
    ) and not args.collect_probe_features:
        logging.info(
            "Enabling --collect_probe_features because VIB/LRP features "
            "are stored inside each low-temperature generation."
        )
        args.collect_probe_features = True

    if getattr(args, "collect_vib_probe", False) and args.compute_lang_align:
        # The optional blank-image chain invokes every o_proj a second time and
        # would make a generic hook capture the wrong branch.  VIB uses only the
        # actual image-conditioned decoding path from the paper.
        if not args.no_lang_align_no_image_chain:
            logging.info(
                "Disabling the lang-align no-image chain while collecting VIB "
                "head outputs so hooks remain aligned to the primary decode."
            )
            args.no_lang_align_no_image_chain = True

    if getattr(args, "vib_mitigation_checkpoint", None):
        if args.compute_lang_align:
            raise ValueError(
                "Online VIB mitigation and lang-align custom decoding cannot run "
                "in the same generation call; use a separate mitigation run."
            )

    # Setup run.
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    experiment_details = {'args': args}
    random.seed(args.random_seed)
    user = os.getenv('USER', 'ceor')
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', 'experiments')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    wandb.init(
        entity=args.entity,
        project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')

    # Get accuracy metric.
    metric = utils.get_metric_vlm(args.metric)

    # Load dataset.
    validation_only = (
        not args.get_training_set_generations
        and not args.compute_p_true
        and int(args.num_few_shot) == 0
        and args.ood_train_dataset is None
    )
    train_dataset, validation_dataset = load_ds(
        args.dataset,
        add_options=args.use_mc_options,
        seed=args.random_seed,
        validation_only=validation_only,
        pr_bench_train_fraction=getattr(args, "pr_bench_train_fraction", 0.0),
    )
    if validation_only:
        logging.info(
            "Validation-only dataset loading enabled; training annotations are skipped."
        )
    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        # Get indices of answerable and unanswerable questions and construct prompt.
        train_dataset, _ = load_ds(
            args.ood_train_dataset,
            seed=args.random_seed,
            add_options=args.use_mc_options,
            pr_bench_train_fraction=getattr(args, "pr_bench_train_fraction", 0.0),
        )
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions and construct prompt.
    if args.dataset in VLM_DATASETS:
        answerable_indices, unanswerable_indices = utils.split_vlm_dataset(train_dataset)
    else:
        answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = _sample_indices(
        answerable_indices,
        args.num_few_shot,
        label="few-shot prompt",
    )
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)# 是否需要加入和image
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    # prompt = utils.construct_fewshot_prompt_from_indices(
    #     train_dataset, prompt_indices, BRIEF, arg, make_prompt)

    prompt = utils.construct_fewshot_prompt_from_indices_multimodal(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt, "image")

    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Model initialization is deferred until after checkpoint discovery.  A
    # reference-only resume with complete target/p_true shards does not need to
    # allocate the target VLM or construct its external feature parsers.
    model = None
    reference_ensemble = None
    semantic_uq_encoder = None
    use_lang_align = False

    # Select p_true demonstrations before freezing the train/validation plans.
    # Their expensive model responses are checkpointed after the run identity is
    # known below.
    p_true_indices = []
    if args.compute_p_true:
        p_true_indices = _sample_indices(
            answerable_indices,
            args.p_true_num_fewshot,
            label="p_true few-shot",
        )
        if not p_true_indices:
            logging.warning(
                "No answerable training examples are available for p_true; "
                "using a zero-shot p_true prompt. Provide --ood_train_dataset "
                "if p_true demonstrations are required."
            )
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))

    # Freeze both split selections before making any auxiliary API request.  The
    # resulting sample IDs are part of the checkpoint fingerprint, so re-running
    # the same command selects and resumes exactly the same work.
    split_plan = _select_generation_splits(
        args,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        remaining_answerable=remaining_answerable,
        unanswerable_indices=unanswerable_indices,
    )
    for dataset_split, split_info in split_plan.items():
        experiment_details[dataset_split] = {"indices": split_info["indices"]}
    split_sample_ids = {
        dataset_split: [
            split_info["dataset"][index]["question_id"]
            for index in split_info["indices"]
        ]
        for dataset_split, split_info in split_plan.items()
    }
    if p_true_indices:
        split_sample_ids["p_true_few_shot"] = [
            train_dataset[index]["question_id"] for index in p_true_indices
        ]
    reference_config_identity = None
    if getattr(args, "collect_lgd_uq", False):
        with open(args.lgd_reference_config, "r", encoding="utf-8") as handle:
            reference_config_identity = _reference_config_checkpoint_identity(
                json.load(handle)
            )
    checkpoint_fingerprint, checkpoint_identity = generation_fingerprint(
        vars(args),
        split_sample_ids=split_sample_ids,
        reference_config=reference_config_identity,
    )
    checkpoint_store = GenerationCheckpointStore(
        args.generation_checkpoint_dir,
        fingerprint=checkpoint_fingerprint,
        identity=checkpoint_identity,
        resume=bool(args.resume_generation),
    )
    experiment_details["generation_checkpoint"] = {
        "root": str(checkpoint_store.root.resolve()),
        "fingerprint": checkpoint_fingerprint,
        "resume": bool(args.resume_generation),
        "format": "atomic_per_sample_target_and_per_model_reference_shards",
    }
    logging.info(
        "Generation checkpoint root: %s (resume=%s).",
        checkpoint_store.root,
        bool(args.resume_generation),
    )

    reference_only_resume = _can_reference_only_resume(
        args, checkpoint_store, split_sample_ids
    )
    if reference_only_resume:
        logging.info(
            "Reference-only resume enabled: all target and p_true shards are "
            "cached; skipping target VLM/GPU and feature-parser initialization."
        )
    else:
        model = utils.init_model(args)

    if getattr(args, "collect_lgd_uq", False):
        reference_ensemble = load_reference_ensemble(
            args.lgd_reference_config,
            target_model_name=getattr(model, "model_name", args.model_name),
        )
        experiment_details["lgd_uq"] = {
            "reference_ensemble": reference_ensemble.describe(),
            "num_reference_samples": int(args.lgd_num_reference_samples),
            "num_target_samples": int(args.lgd_num_target_samples),
            "reference_temperature": float(args.lgd_reference_temperature),
            "target_temperature": float(args.temperature),
            "splits": list(args.lgd_splits),
        }
        logging.info(
            "LGD-UQ teacher enabled with %d reference models (%d draws/model, "
            "%d target draws).",
            len(reference_ensemble.entries),
            int(args.lgd_num_reference_samples),
            int(args.lgd_num_target_samples),
        )
        if _cross_model_semantic_uq_enabled(args):
            semantic_uq_encoder = SentenceT5SimilarityEncoder(
                CrossModelSemanticUQConfig(
                    encoder_name=args.cross_model_semantic_encoder,
                    device=args.cross_model_semantic_device,
                    batch_size=int(args.cross_model_semantic_batch_size),
                )
            )
            experiment_details["cross_model_semantic_uq"] = {
                **semantic_uq_encoder.describe(),
                "enabled_automatically_with_lgd": (
                    getattr(args, "collect_cross_model_semantic_uq", None) is None
                ),
                "method": (
                    "complementing_self_consistency_with_cross_model_disagreement"
                ),
                "auxiliary_model_aggregation": "uniform_paper_equation",
                "outputs": {
                    "AU": "u_aleatoric",
                    "EU": "u_epistemic",
                    "TU": "u_total",
                },
            }
            logging.info(
                "Cross-model semantic UQ enabled with encoder %r on %s.",
                args.cross_model_semantic_encoder,
                args.cross_model_semantic_device,
            )

    if not reference_only_resume:
        if getattr(args, "collect_icr_probe", False) and not hasattr(
            model, "_icr_features_from_sequences"
        ):
            raise ValueError(
                "--collect_icr_probe currently requires the Qwen VLM wrapper. "
                "The standalone ICRScore API can be used with dense internals from "
                "other Hugging Face causal LMs."
            )
        use_lang_align = (
            args.compute_lang_align
            and getattr(model, 'is_multimodal', False)
            and hasattr(model, 'predict_with_lang_align')
            and model._lang_align_runner is not None
        )
        if args.compute_lang_align and not use_lang_align:
            logging.warning(
                'compute_lang_align is enabled but lang-align sampling is unavailable '
                'for model `%s`. Skipping lang-align uncertainty.',
                args.model_name,
            )
    reference_prefetch_handle = None
    if reference_ensemble is not None:
        reference_jobs = _make_lgd_reference_jobs(
            args, split_plan, make_prompt, BRIEF
        )
        reference_prefetch_handle = _LGDReferencePrefetchHandle(
            args=args,
            reference_ensemble=reference_ensemble,
            jobs=reference_jobs,
            checkpoint_store=checkpoint_store,
        )
        _ACTIVE_LGD_PREFETCH_HANDLE = reference_prefetch_handle
        experiment_details["lgd_uq"]["pipeline"] = {
            "target_and_reference_production": "concurrent",
            "aggregation": "after_producers_join",
            "model_workers": int(args.lgd_parallel_workers),
            "requests_per_remote_model": int(args.lgd_requests_per_model),
            "reference_cache_layout": "per_model_per_sample",
        }
        logging.info(
            "LGD reference producers are running in the background while the "
            "target model generates its own sample shards."
        )

    if args.compute_p_true:
        logging.info(80*'#')
        p_true_setup = checkpoint_store.load_state("p_true_setup")
        if isinstance(p_true_setup, Mapping):
            p_true_few_shot_prompt = p_true_setup["prompt"]
            p_true_responses = p_true_setup["responses"]
            len_p_true = int(p_true_setup["length"])
            logging.info(
                "Resumed checkpointed p_true few-shot prompt (%d examples).",
                len_p_true,
            )
        else:
            logging.info('Constructing few-shot prompt for p_true.')
            # p_true benefits substantially from few-shot model responses.
            p_true_few_shot_prompt, p_true_responses, len_p_true = (
                p_true_utils.construct_few_shot_prompt_vlm(
                    model=model,
                    dataset=train_dataset,
                    indices=p_true_indices,
                    prompt=prompt,
                    brief=BRIEF,
                    brief_always=args.brief_always and args.enable_brief,
                    make_prompt=make_prompt,
                    num_generations=args.num_generations,
                    metric=metric,
                )
            )
            checkpoint_store.save_state("p_true_setup", {
                "prompt": p_true_few_shot_prompt,
                "responses": p_true_responses,
                "length": int(len_p_true),
            })
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))
        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []
        grounding_eval_records = []
        lgd_u_as, lgd_u_ds, lgd_u_es, lgd_u_totals = [], [], [], []
        lgd_iou_ablation_measures = defaultdict(lambda: defaultdict(list))
        cross_model_semantic_measures = defaultdict(list)

        def accumulate_resumed_record(record):
            most_likely = record.get("most_likely_answer", {})
            if "accuracy" in most_likely:
                accuracies.append(float(most_likely["accuracy"]))
            grounding_eval = most_likely.get("grounding_eval")
            if isinstance(grounding_eval, Mapping):
                grounding_eval_records.append(dict(grounding_eval))
            lgd_result = record.get("lgd_uq")
            if isinstance(lgd_result, Mapping):
                uncertainty = lgd_result.get("uncertainty", {})
                for key, destination in (
                    ("u_a", lgd_u_as),
                    ("u_d", lgd_u_ds),
                    ("u_e", lgd_u_es),
                    ("u_total", lgd_u_totals),
                ):
                    destination.append(float(uncertainty[key]))
                ablation = lgd_result.get("iou_threshold_ablation", {})
                for threshold_key, threshold_result in ablation.get(
                    "results", {}
                ).items():
                    threshold_uncertainty = threshold_result["uncertainty"]
                    for metric in ("u_a", "u_d", "u_e", "u_total"):
                        lgd_iou_ablation_measures[threshold_key][metric].append(
                            float(threshold_uncertainty[metric])
                        )
            semantic_result = record.get("cross_model_semantic_uq")
            if isinstance(semantic_result, Mapping):
                semantic_uncertainty = semantic_result.get("uncertainty", {})
                for metric in ("u_aleatoric", "u_epistemic", "u_total"):
                    cross_model_semantic_measures[metric].append(
                        float(semantic_uncertainty[metric])
                    )
            if "p_true" in record:
                p_trues.append(float(record["p_true"]))

        if dataset_split not in split_plan:
            logging.info('Skip %s data.', dataset_split)
            continue
        dataset = split_plan[dataset_split]["dataset"]
        indices = split_plan[dataset_split]["indices"]

        cached_generations = {}
        for sample_id in split_sample_ids[dataset_split]:
            cached = checkpoint_store.load_generation(dataset_split, sample_id)
            if cached is not None:
                cached_generations[sample_id] = cached
        missing_target_count = len(indices) - len(cached_generations)
        logging.info(
            "%s target checkpoint status: %d/%d samples cached; %d require "
            "target-model generation.",
            dataset_split,
            len(cached_generations),
            len(indices),
            missing_target_count,
        )

        it = 0
        for index, sample_id in tqdm(
            zip(indices, split_sample_ids[dataset_split]),
            total=len(indices),
            desc=(
                f"{dataset_split}: {len(cached_generations)} cached, "
                f"{missing_target_count} to generate"
            ),
        ):
            if (it + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            cached_generation = cached_generations.get(sample_id)
            if cached_generation is not None:
                generations[sample_id] = cached_generation
                accumulate_resumed_record(cached_generation)
                logging.debug(
                    "Resumed completed %s sample %s (%d/%d).",
                    dataset_split,
                    sample_id,
                    it,
                    len(indices),
                )
                continue

            # Decode the dataset row/image only when target generation is missing.
            example = dataset[index]
            question, context = example["question"], None

            # Optional image for multimodal models (e.g. Qwen3-VL).
            image: Any = _example_image(example, open_path=True)

            # Store `image` for later embedding-based metrics (e.g. cosine similarity
            # between image+question and generated answers).
            generations[example['question_id']] = {
                'question': question,
                'context': context,
                'image': image,
                'sample_metadata': _sample_metadata(example, image),
            }
            if "answer" in example:
                correct_answer = example["answer"]
            elif "answers" in example:
                if example["answers"] is None:
                    correct_answer = ""
                elif isinstance(example['answers'][0], dict):
                    correct_answer = example['answers'][0]["answer"]
                else:
                    correct_answer = example['answers'][0]

            current_input = make_prompt(
                context, question, None, BRIEF, args.brief_always and args.enable_brief)
            # local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []

            # We sample one low temperature answer on which we will compute the
            # accuracy and args.num_generation high temperature answers which will
            # be used to estimate the entropy variants.

            collect_lgd_here = (
                getattr(args, "collect_lgd_uq", False)
                and dataset_split in set(args.lgd_splits)
            )
            if (
                dataset_split == 'train'
                and args.get_training_set_generations_most_likely_only
                and not collect_lgd_here
            ):
                num_generations = 1
            else:
                high_temperature_generations = int(args.num_generations)
                if collect_lgd_here:
                    high_temperature_generations = max(
                        high_temperature_generations,
                        int(args.lgd_num_target_samples),
                    )
                num_generations = high_temperature_generations + 1

            for i in range(num_generations):

                # The primary answer feeds task accuracy and all paper probes.
                temperature = args.low_temperature if i == 0 else args.temperature

                # For multimodal models (e.g. Qwen3-VL) we transparently pass the
                # image if the model supports it and the example provides one.
                predict_kwargs = {}
                if getattr(model, "is_multimodal", False) and image is not None:
                    predict_kwargs["image"] = image

                lang_align_metrics = None
                probe_features = None
                vib_mitigation = None
                collect_lang_align_here = (
                    i == 0
                    and use_lang_align
                    and image is not None
                    and dataset_split == 'validation'
                )
                if (
                    i == 0
                    and getattr(args, "vib_mitigation_checkpoint", None)
                    and hasattr(model, "predict_with_vib_mitigation")
                ):
                    (
                        predicted_answer,
                        token_log_likelihoods,
                        embedding,
                        probe_features,
                        vib_mitigation,
                    ) = model.predict_with_vib_mitigation(
                        current_input, temperature, **predict_kwargs
                    )
                elif (
                    collect_lang_align_here
                    and args.collect_probe_features
                    and hasattr(model, 'predict_with_lang_align_probe')
                ):
                    (
                        predicted_answer,
                        token_log_likelihoods,
                        embedding,
                        lang_align_metrics,
                        probe_features,
                    ) = model.predict_with_lang_align_probe(
                        current_input, temperature, **predict_kwargs)
                elif (
                    collect_lang_align_here
                ):
                    predicted_answer, token_log_likelihoods, embedding, lang_align_metrics = (
                        model.predict_with_lang_align(
                            current_input, temperature, **predict_kwargs)
                    )
                elif (
                    i == 0
                    and args.collect_probe_features
                    and hasattr(model, 'predict_with_probe')
                ):
                    predicted_answer, token_log_likelihoods, embedding, probe_features = (
                        model.predict_with_probe(
                            current_input, temperature, **predict_kwargs)
                    )
                else:
                    predicted_answer, token_log_likelihoods, embedding = model.predict(
                        current_input, temperature, **predict_kwargs)
                embedding = embedding.cpu() if embedding is not None else None

                # Only compute accuracy if question is answerable.
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                grounding_eval = None
                if correct_answer and compute_acc and args.metric == 'grounding':
                    grounding_eval = utils.grounding_u.evaluate_grounding_prediction(
                        predicted_answer,
                        example,
                    )
                    acc = grounding_eval["accuracy"]
                elif correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0  # pylint: disable=invalid-name

                if i == 0:
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                    if args.use_context:
                        logging.info('context: '.ljust(15) + str(context))
                    logging.info('question: '.ljust(15) + question)
                    logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                    logging.info('correct answer: '.ljust(15) + str(correct_answer))
                    logging.info('accuracy: '.ljust(15) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'prompt': current_input,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        'accuracy': acc}
                    if probe_features is not None:
                        most_likely_answer_dict['probe_features'] = probe_features
                        if 'generated_token_ids' in probe_features:
                            most_likely_answer_dict['generated_token_ids'] = probe_features['generated_token_ids']
                            most_likely_answer_dict['prompt'] = current_input
                            most_likely_answer_dict['generation_temperature'] = float(temperature)
                        logging.info(
                            'probe_argus_h_dim: '.ljust(15) + str(
                                int(probe_features['argus_h_mean'].numel())))
                        logging.info(
                            'probe_num_generated_tokens: '.ljust(15) + str(
                                probe_features['num_generated_tokens']))
                    if vib_mitigation is not None:
                        most_likely_answer_dict['vib_mitigation'] = vib_mitigation
                        logging.info(
                            'vib_mitigation_triggers: '.ljust(15) + str(
                                vib_mitigation['num_triggered_tokens']))
                    if args.metric == 'grounding':
                        if grounding_eval is None:
                            grounding_eval = utils.grounding_u.evaluate_grounding_prediction(
                                predicted_answer,
                                example,
                            )
                        most_likely_answer_dict['grounding_eval'] = grounding_eval
                        grounding_eval_records.append(grounding_eval)
                        logging.info('grounding_iou: '.ljust(15) + str(grounding_eval['iou']))
                        logging.info('grounding_accuracy: '.ljust(15) + str(grounding_eval['accuracy']))
                    if lang_align_metrics is not None:
                        # Preserve attention diagnostics without persisting
                        # retired uncertainty scores.
                        most_likely_answer_dict['lang_align_uncertainty'] = {
                            name: value for name, value in lang_align_metrics.items()
                            if not is_retired_uncertainty_measure(name)
                        }
                    generations[example['question_id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        # 'reference': utils.get_reference(example)})
                        'answer': str(correct_answer),
                        'reference': ""})####################################################

                else:
                    logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, acc))

            # Append all predictions for this example to `generations`.
            generations[example['question_id']]['responses'] = full_responses

            if collect_lgd_here:
                logging.debug(
                    "Saved target shard for %s/%s; LGD and semantic UQ are "
                    "deferred until the background reference producers join.",
                    dataset_split,
                    example["question_id"],
                )

            if args.compute_p_true and dataset_split == 'validation':
                # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                p_true = p_true_utils.calculate_p_true(
                    model, question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], p_true_few_shot_prompt,
                    hint=args.p_true_hint)
                p_trues.append(p_true)
                generations[example['question_id']]["p_true"] = float(p_true)
                logging.info('p_true: %s', p_true)

            checkpoint_store.save_generation(
                dataset_split,
                example["question_id"],
                generations[example["question_id"]],
            )

        # Older checkpoint shards can still contain retired scores.
        for record in generations.values():
            most_likely = record.get('most_likely_answer', {})
            most_likely.pop('visual_trajectory', None)
            metrics = most_likely.get('lang_align_uncertainty')
            if isinstance(metrics, Mapping):
                most_likely['lang_align_uncertainty'] = {
                    name: value for name, value in metrics.items()
                    if not is_retired_uncertainty_measure(name)
                }
            evidence = most_likely.get('probe_evidence_features')
            if isinstance(evidence, Mapping):
                remaining_evidence = {
                    name: value for name, value in evidence.items()
                    if not is_retired_uncertainty_measure(name)
                }
                if remaining_evidence:
                    most_likely['probe_evidence_features'] = remaining_evidence
                else:
                    most_likely.pop('probe_evidence_features', None)

        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')
        if (
            reference_ensemble is not None
            and dataset_split in set(args.lgd_splits)
        ):
            checkpoint_store.save_state(
                f"target_generation_{dataset_split}_complete",
                {"num_samples": len(generations)},
            )
        else:
            checkpoint_store.mark_split_complete(
                dataset_split, num_samples=len(generations)
            )

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        for threshold_key in sorted(lgd_iou_ablation_measures, key=float):
            values = lgd_iou_ablation_measures[threshold_key]
            print(
                f"LGD-UQ {dataset_split} IoU={threshold_key} mean: "
                f"U_A={np.mean(values['u_a']):.6f} "
                f"U_D={np.mean(values['u_d']):.6f} "
                f"U_E={np.mean(values['u_e']):.6f} "
                f"U_T={np.mean(values['u_total']):.6f}"
            )
        if cross_model_semantic_measures:
            semantic_means = {
                metric: float(np.mean(values))
                for metric, values in cross_model_semantic_measures.items()
            }
            print(
                f"Original-paper cross-model semantic UQ {dataset_split} mean: "
                f"AU={semantic_means['u_aleatoric']:.6f} "
                f"EU={semantic_means['u_epistemic']:.6f} "
                f"TU={semantic_means['u_total']:.6f}"
            )
            experiment_details[dataset_split][
                "cross_model_semantic_uq_summary"
            ] = {
                "num_samples": len(cross_model_semantic_measures["u_total"]),
                "mean_au": semantic_means["u_aleatoric"],
                "mean_eu": semantic_means["u_epistemic"],
                "mean_tu": semantic_means["u_total"],
            }
            wandb.log({
                f"{dataset_split}_cross_model_semantic_au_mean": semantic_means[
                    "u_aleatoric"
                ],
                f"{dataset_split}_cross_model_semantic_eu_mean": semantic_means[
                    "u_epistemic"
                ],
                f"{dataset_split}_cross_model_semantic_tu_mean": semantic_means[
                    "u_total"
                ],
            })
        wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == 'validation':
            uncertainty_measures = {}
            if args.compute_p_true:
                uncertainty_measures.update({
                    'p_false':  [1 - p for p in p_trues],
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
                })
            if lgd_u_as:
                uncertainty_measures.update({
                    "lgd_u_a": lgd_u_as,
                    "lgd_u_d": lgd_u_ds,
                    "lgd_u_e": lgd_u_es,
                    "lgd_u_total": lgd_u_totals,
                })
                for threshold_key, threshold_values in (
                    lgd_iou_ablation_measures.items()
                ):
                    safe_threshold = threshold_key.replace(".", "_")
                    for metric, values in threshold_values.items():
                        uncertainty_measures[
                            f"lgd_iou_{safe_threshold}_{metric}"
                        ] = values
            if cross_model_semantic_measures:
                uncertainty_measures.update({
                    "cross_model_semantic_au": cross_model_semantic_measures[
                        "u_aleatoric"
                    ],
                    "cross_model_semantic_eu": cross_model_semantic_measures[
                        "u_epistemic"
                    ],
                    "cross_model_semantic_tu": cross_model_semantic_measures[
                        "u_total"
                    ],
                })
            if uncertainty_measures:
                uncertainty_measures = filter_reportable_uncertainty_measures(
                    uncertainty_measures
                )
            if uncertainty_measures:
                results_dict['uncertainty_measures'] = uncertainty_measures
            if grounding_eval_records:
                results_dict['grounding_eval'] = grounding_eval_records
            if uncertainty_measures or grounding_eval_records:
                utils.save(results_dict, 'uncertainty_measures.pkl')

    # No target-model calls occur after this point.  Release its GPU allocation
    # before sentence-T5 is lazily loaded for semantic aggregation.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if reference_prefetch_handle is not None:
        logging.info(
            "Target production is complete; waiting only for unfinished LGD "
            "reference shards before final aggregation."
        )
        reference_prefetch_handle.wait()
        _ACTIVE_LGD_PREFETCH_HANDLE = None
        # Reference adapters are no longer needed: finalization reads only their
        # independent cache shards and normalized ensemble weights.
        reference_ensemble.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _finalize_lgd_pipeline(
            args=args,
            split_plan=split_plan,
            checkpoint_store=checkpoint_store,
            reference_ensemble=reference_ensemble,
            semantic_uq_encoder=semantic_uq_encoder,
            make_prompt=make_prompt,
            brief=BRIEF,
            experiment_details=experiment_details,
        )

    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    if semantic_uq_encoder is not None:
        semantic_uq_encoder.close()


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')


    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    try:
        main(args)
    finally:
        try:
            if _ACTIVE_LGD_PREFETCH_HANDLE is not None:
                logging.info(
                    "Stopping background LGD producers after an interrupted run; "
                    "completed target/reference shards remain resumable."
                )
                _ACTIVE_LGD_PREFETCH_HANDLE.stop()
        finally:
            if getattr(wandb, "run", None) is not None:
                logging.info("Finalizing W&B run.")
                wandb.finish()
    logging.info('FINISHED `generate_answers`!')

    # if args.compute_uncertainties:
    #     # Follow with uncertainty calculation script by default.
    #     args.assign_new_wandb_id = False
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     logging.info(50 * '#X')
    #     logging.info('STARTING `compute_uncertainty_measures`!')
    #     main_compute(args)
    #     logging.info('FINISHED `compute_uncertainty_measures`!')
