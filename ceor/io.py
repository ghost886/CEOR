"""Small, model-independent I/O helpers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def read_jsonl(path):
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if not sample_id or sample_id in seen:
                raise ValueError(f"Empty/duplicate sample_id at {path}:{line_number}")
            seen.add(sample_id)
            yield {**row, "sample_id": sample_id}


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def image_digest(image):
    image = image.convert("RGB")
    digest = hashlib.sha256(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()
