"""Crash-safe, per-sample checkpoints for long generation jobs.

The generation artifacts produced by this project can be several gigabytes, so
rewriting one monolithic pickle after every example is prohibitively expensive.
This module stores small atomic shards instead.  Auxiliary-model shards are
further separated by model, which lets reference workers run and resume
independently.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import re
import shutil
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence


CHECKPOINT_SCHEMA_VERSION = 1
LGD_REFERENCE_SIDECAR_SCHEMA_VERSION = 1
LGD_REFERENCE_SIDECAR_KIND = "lgd_reference_samples_by_split"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=str)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def generation_fingerprint(
    args: Mapping[str, Any],
    *,
    split_sample_ids: Mapping[str, Sequence[Any]],
    reference_config: Optional[Mapping[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Return a stable identity for outputs affected by the current run config."""
    operational_only = {
        "assign_new_wandb_id",
        "debug",
        "entity",
        "generation_checkpoint_dir",
        "lgd_parallel_workers",
        "lgd_requests_per_model",
        "resume_generation",
    }
    output_args = {
        key: value for key, value in args.items() if key not in operational_only
    }
    identity = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "args": _jsonable(output_args),
        "split_sample_ids": _jsonable(split_sample_ids),
        "reference_config": _jsonable(reference_config),
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), identity


def _safe_segment(value: Any, *, fallback: str) -> str:
    raw = str(value).strip()
    compact = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return (compact[:72] or fallback)


def _sample_token(sample_id: Any) -> str:
    encoded = json.dumps(
        _jsonable(sample_id), ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    prefix = _safe_segment(sample_id, fallback="sample")[:36]
    return f"{prefix}-{digest}"


def _reference_model_segment(model_name: str) -> str:
    model_segment = _safe_segment(model_name, fallback="model")
    model_digest = hashlib.sha256(str(model_name).encode("utf-8")).hexdigest()[:10]
    return f"{model_segment}-{model_digest}"


def _component_base_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Identity shared by target generation and every reference producer."""
    return {
        "schema_version": identity.get("schema_version"),
        "args": identity.get("args"),
        "split_sample_ids": identity.get("split_sample_ids"),
    }


def _reference_config_parts(
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = identity.get("reference_config")
    if not isinstance(config, Mapping):
        return {}, {}
    global_config = {
        str(key): value for key, value in config.items() if key != "models"
    }
    models = {}
    raw_models = config.get("models")
    if isinstance(raw_models, list):
        for model in raw_models:
            if isinstance(model, Mapping) and model.get("name"):
                models[str(model["name"])] = dict(model)
    return global_config, models


def _link_tree_missing(source_root: Path, destination_root: Path) -> int:
    """Hard-link compatible immutable shards, copying only as a fallback."""
    if not source_root.is_dir():
        return 0
    imported = 0
    for source in source_root.rglob("*"):
        if not source.is_file() or source.name.startswith("."):
            continue
        destination = destination_root / source.relative_to(source_root)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except FileExistsError:
            continue
        except OSError:
            if destination.exists():
                continue
            shutil.copy2(source, destination)
        imported += 1
    return imported


def _inherit_reference_tree(source_root: Path, destination_root: Path) -> int:
    """Import missing shards and replace only shorter compatible draw prefixes."""
    if not source_root.is_dir():
        return 0
    imported = 0
    for source in source_root.glob("*.pkl"):
        destination = destination_root / source.name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except FileExistsError:
                continue
            except OSError:
                if destination.exists():
                    continue
                shutil.copy2(source, destination)
            imported += 1
            continue

        source_payload = _load_pickle(source)
        destination_payload = _load_pickle(destination)
        if not isinstance(source_payload, Mapping):
            continue
        if not isinstance(destination_payload, Mapping):
            _atomic_pickle_dump(source_payload, destination)
            imported += 1
            continue
        if source_payload.get("sample_id") != destination_payload.get("sample_id"):
            continue
        if source_payload.get("model_name") != destination_payload.get("model_name"):
            continue
        source_records = source_payload.get("records")
        destination_records = destination_payload.get("records")
        if not isinstance(source_records, list):
            continue
        if not isinstance(destination_records, list) or len(source_records) > len(
            destination_records
        ):
            _atomic_pickle_dump(source_payload, destination)
            imported += 1
    return imported


def _atomic_pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (EOFError, OSError, pickle.PickleError) as exc:
        logging.warning("Ignoring unreadable generation checkpoint %s: %s", path, exc)
        return None


def build_lgd_reference_sidecar(
    *,
    split: str,
    sample_ids: Sequence[Any],
    model_names: Sequence[str],
    num_draws_per_model: int,
    references_by_sample: Mapping[Any, Mapping[str, Sequence[Mapping[str, Any]]]],
    checkpoint_root: Optional[str] = None,
) -> dict[str, Any]:
    """Create the portable LGD file written beside ``*_generations.pkl``."""
    ordered_sample_ids = list(sample_ids)
    ordered_model_names = [str(name) for name in model_names]
    requested_draws = int(num_draws_per_model)
    if requested_draws < 1:
        raise ValueError("LGD sidecar num_draws_per_model must be positive.")
    samples = {}
    for sample_id in ordered_sample_ids:
        if sample_id not in references_by_sample:
            raise ValueError(
                f"LGD sidecar is missing split {split!r} sample {sample_id!r}."
            )
        model_records = references_by_sample[sample_id]
        missing_models = [
            name for name in ordered_model_names if name not in model_records
        ]
        if missing_models:
            raise ValueError(
                f"LGD sidecar sample {sample_id!r} is missing models "
                f"{missing_models}."
            )
        samples[sample_id] = {
            name: [dict(record) for record in model_records[name]]
            for name in ordered_model_names
        }
    payload = {
        "schema_version": LGD_REFERENCE_SIDECAR_SCHEMA_VERSION,
        "kind": LGD_REFERENCE_SIDECAR_KIND,
        "split": str(split),
        "sample_ids": ordered_sample_ids,
        "model_names": ordered_model_names,
        "num_draws_per_model": requested_draws,
        "samples": samples,
    }
    if checkpoint_root is not None:
        payload["source_checkpoint_root"] = str(checkpoint_root)
    validate_lgd_reference_sidecar(
        payload,
        expected_split=split,
        expected_sample_ids=ordered_sample_ids,
        expected_model_names=ordered_model_names,
    )
    return payload


def validate_lgd_reference_sidecar(
    payload: Any,
    *,
    expected_split: Optional[str] = None,
    expected_sample_ids: Optional[Sequence[Any]] = None,
    expected_model_names: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Strictly validate sample/model/draw alignment and return a summary."""
    if not isinstance(payload, Mapping):
        raise ValueError("LGD reference sidecar must be a mapping.")
    if payload.get("schema_version") != LGD_REFERENCE_SIDECAR_SCHEMA_VERSION:
        raise ValueError("Unsupported LGD reference sidecar schema version.")
    if payload.get("kind") != LGD_REFERENCE_SIDECAR_KIND:
        raise ValueError("Invalid LGD reference sidecar kind.")
    split = payload.get("split")
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"LGD reference sidecar split mismatch: {split!r} != "
            f"{expected_split!r}."
        )
    sample_ids = payload.get("sample_ids")
    model_names = payload.get("model_names")
    samples = payload.get("samples")
    if not isinstance(sample_ids, list) or not isinstance(model_names, list):
        raise ValueError("LGD reference sidecar IDs and model names must be lists.")
    if not isinstance(samples, Mapping):
        raise ValueError("LGD reference sidecar samples must be a mapping.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("LGD reference sidecar contains duplicate sample IDs.")
    if len(model_names) != len(set(model_names)):
        raise ValueError("LGD reference sidecar contains duplicate model names.")
    if expected_sample_ids is not None and sample_ids != list(expected_sample_ids):
        raise ValueError(
            "LGD reference sidecar sample order does not match generations.pkl."
        )
    if (
        expected_model_names is not None
        and model_names != [str(name) for name in expected_model_names]
    ):
        raise ValueError("LGD reference sidecar model order does not match config.")
    if set(samples) != set(sample_ids):
        raise ValueError("LGD reference sidecar sample keys do not match sample_ids.")
    requested_draws = int(payload.get("num_draws_per_model", 0))
    if requested_draws < 1:
        raise ValueError("LGD reference sidecar draw count must be positive.")
    for sample_id in sample_ids:
        model_records = samples[sample_id]
        if not isinstance(model_records, Mapping):
            raise ValueError(
                f"LGD reference sidecar sample {sample_id!r} must be a mapping."
            )
        if set(model_records) != set(model_names):
            raise ValueError(
                f"LGD reference sidecar model keys mismatch for {sample_id!r}."
            )
        for model_name in model_names:
            records = model_records[model_name]
            if not isinstance(records, list) or len(records) != requested_draws:
                raise ValueError(
                    f"LGD reference sidecar {split!r}/{sample_id!r}/"
                    f"{model_name!r} has {len(records) if isinstance(records, list) else 0}/"
                    f"{requested_draws} draws."
                )
            for expected_draw, record in enumerate(records):
                if not isinstance(record, Mapping) or record.get("draw") != expected_draw:
                    raise ValueError(
                        f"LGD reference sidecar {split!r}/{sample_id!r}/"
                        f"{model_name!r} is not a contiguous draw sequence."
                    )
    return {
        "schema_version": LGD_REFERENCE_SIDECAR_SCHEMA_VERSION,
        "kind": LGD_REFERENCE_SIDECAR_KIND,
        "split": split,
        "num_samples": len(sample_ids),
        "model_names": list(model_names),
        "num_models": len(model_names),
        "num_draws_per_model": requested_draws,
        "source_checkpoint_root": payload.get("source_checkpoint_root"),
    }


class GenerationCheckpointStore:
    """Filesystem layout and validation for resumable generation shards."""

    def __init__(
        self,
        parent_dir: str | os.PathLike[str],
        *,
        fingerprint: str,
        identity: Mapping[str, Any],
        resume: bool = True,
    ) -> None:
        base = Path(parent_dir).expanduser()
        # A directory containing a manifest is an explicit run root.  This also
        # makes a previously interrupted ``-fresh-...`` run recoverable by passing
        # its exact path as --generation_checkpoint_dir.
        explicit_run_root = resume and (base / "manifest.json").is_file()
        if explicit_run_root:
            self.root = base
        else:
            root_name = fingerprint[:16]
            if not resume:
                root_name = f"{root_name}-fresh-{int(time.time())}-{os.getpid()}"
            self.root = base / root_name
        self.fingerprint = fingerprint
        self.identity = _jsonable(identity)
        self.resume = bool(resume)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists() and self.resume:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot read generation checkpoint manifest {manifest_path}: {exc}"
                ) from exc
            if manifest.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    "Generation checkpoint fingerprint mismatch; choose another "
                    "--generation_checkpoint_dir or run with --no-resume_generation."
                )
        else:
            _atomic_json_dump({
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "identity": self.identity,
            }, manifest_path)
        if self.resume:
            self._inherit_compatible_components()

    def _compatible_sibling_runs(self) -> list[tuple[Path, dict[str, Any]]]:
        """Find prior runs differing only in independently cached components."""
        shared_identity = _component_base_identity(self.identity)
        candidates = []
        try:
            siblings = list(self.root.parent.iterdir())
        except OSError:
            return candidates
        for candidate in siblings:
            if candidate == self.root or not candidate.is_dir():
                continue
            manifest_path = candidate / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            candidate_identity = manifest.get("identity")
            if not isinstance(candidate_identity, Mapping):
                continue
            if _component_base_identity(candidate_identity) != shared_identity:
                continue
            candidates.append((candidate, dict(candidate_identity)))
        return candidates

    def _inherit_compatible_components(self) -> None:
        """Reuse target and unchanged-model shards across reference config edits.

        The aggregate fingerprint intentionally changes when any reference model
        changes.  Target generations and other reference models are independent,
        though, so copying their immutable per-sample shards into the new run is
        both safe and substantially cheaper than regenerating everything.
        """
        candidates = self._compatible_sibling_runs()
        if not candidates:
            return

        target_source, _ = max(
            candidates,
            key=lambda item: sum(
                1 for _ in (item[0] / "target_generations").glob("*/*.pkl")
            ),
        )
        imported_targets = _link_tree_missing(
            target_source / "target_generations",
            self.root / "target_generations",
        )
        imported_state = _link_tree_missing(
            target_source / "run_state", self.root / "run_state"
        )

        current_global, current_models = _reference_config_parts(self.identity)
        imported_references = 0
        imported_models = []
        for model_name, model_spec in current_models.items():
            compatible = []
            for candidate_root, candidate_identity in candidates:
                candidate_global, candidate_models = _reference_config_parts(
                    candidate_identity
                )
                if candidate_global != current_global:
                    continue
                if candidate_models.get(model_name) != model_spec:
                    continue
                model_dir = _reference_model_segment(model_name)
                count = sum(
                    1
                    for _ in (candidate_root / "lgd_references").glob(
                        f"*/{model_dir}/*.pkl"
                    )
                )
                compatible.append((count, candidate_root))
            if not compatible:
                continue
            _, reference_source = max(compatible, key=lambda item: item[0])
            model_imported = 0
            model_dir = _reference_model_segment(model_name)
            split_ids = self.identity.get("split_sample_ids", {})
            if isinstance(split_ids, Mapping):
                for split in split_ids:
                    if split == "p_true_few_shot":
                        continue
                    model_imported += _inherit_reference_tree(
                        reference_source
                        / "lgd_references"
                        / _safe_segment(split, fallback="split")
                        / model_dir,
                        self.root
                        / "lgd_references"
                        / _safe_segment(split, fallback="split")
                        / model_dir,
                    )
            if model_imported:
                imported_models.append(model_name)
                imported_references += model_imported

        if imported_targets or imported_state or imported_references:
            logging.info(
                "Inherited compatible checkpoint components into %s: "
                "%d target shards, %d run-state shards, %d reference shards "
                "for models %s. Changed reference models remain uncached.",
                self.root,
                imported_targets,
                imported_state,
                imported_references,
                imported_models,
            )

    def _generation_path(self, split: str, sample_id: Any) -> Path:
        return (
            self.root
            / "target_generations"
            / _safe_segment(split, fallback="split")
            / f"{_sample_token(sample_id)}.pkl"
        )

    def _reference_path(self, split: str, model_name: str, sample_id: Any) -> Path:
        return (
            self.root
            / "lgd_references"
            / _safe_segment(split, fallback="split")
            / _reference_model_segment(model_name)
            / f"{_sample_token(sample_id)}.pkl"
        )

    def _state_path(self, name: str) -> Path:
        return self.root / "run_state" / f"{_safe_segment(name, fallback='state')}.pkl"

    def load_state(self, name: str) -> Any:
        payload = _load_pickle(self._state_path(name))
        if not isinstance(payload, Mapping):
            return None
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return None
        if payload.get("name") != name:
            return None
        return payload.get("value")

    def save_state(self, name: str, value: Any) -> Path:
        path = self._state_path(name)
        _atomic_pickle_dump({
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "name": name,
            "value": value,
        }, path)
        return path

    def load_generation(self, split: str, sample_id: Any) -> Optional[dict[str, Any]]:
        payload = _load_pickle(self._generation_path(split, sample_id))
        if not isinstance(payload, Mapping):
            return None
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return None
        if payload.get("sample_id") != sample_id:
            return None
        record = payload.get("record")
        return dict(record) if isinstance(record, Mapping) else None

    def save_generation(
        self, split: str, sample_id: Any, record: Mapping[str, Any]
    ) -> Path:
        path = self._generation_path(split, sample_id)
        _atomic_pickle_dump({
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sample_id": sample_id,
            "record": dict(record),
        }, path)
        return path

    def load_reference(
        self, split: str, model_name: str, sample_id: Any
    ) -> list[dict[str, Any]]:
        payload = _load_pickle(self._reference_path(split, model_name, sample_id))
        if not isinstance(payload, Mapping):
            return []
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return []
        if payload.get("sample_id") != sample_id:
            return []
        if payload.get("model_name") != model_name:
            return []
        records = payload.get("records")
        if not isinstance(records, list):
            return []
        # Only a contiguous draw prefix is resumable.  This prevents a damaged or
        # manually edited shard from causing later draws to be silently skipped.
        contiguous = []
        for expected_draw, record in enumerate(records):
            if not isinstance(record, Mapping) or record.get("draw") != expected_draw:
                break
            contiguous.append(dict(record))
        return contiguous

    def save_reference(
        self,
        split: str,
        model_name: str,
        sample_id: Any,
        records: Sequence[Mapping[str, Any]],
    ) -> Path:
        path = self._reference_path(split, model_name, sample_id)
        _atomic_pickle_dump({
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sample_id": sample_id,
            "model_name": model_name,
            "records": [dict(record) for record in records],
        }, path)
        return path

    def mark_split_complete(self, split: str, *, num_samples: int) -> Path:
        path = self.root / "completed" / f"{_safe_segment(split, fallback='split')}.json"
        _atomic_json_dump({
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "split": split,
            "num_samples": int(num_samples),
        }, path)
        return path


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "LGD_REFERENCE_SIDECAR_KIND",
    "LGD_REFERENCE_SIDECAR_SCHEMA_VERSION",
    "GenerationCheckpointStore",
    "build_lgd_reference_sidecar",
    "generation_fingerprint",
    "validate_lgd_reference_sidecar",
]
