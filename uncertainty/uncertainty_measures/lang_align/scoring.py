from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def resolve_skip_token_ids(
    tokenizer,
    config_skip: Optional[Sequence[int]],
    config_extra: Optional[Sequence[int]],
) -> torch.Tensor:
    if config_skip is not None:
        ids = list(config_skip)
    else:
        ids = list(getattr(tokenizer, "all_special_ids", []) or [])
        if config_extra:
            ids.extend(config_extra)
    return torch.tensor(sorted(set(ids)), dtype=torch.long)


def pred_probs_from_step_logits(
    step_logits: Sequence[torch.Tensor],
    token_ids: Sequence[int],
) -> torch.Tensor:
    """Per-step probability of realized token ids from generation logits."""
    n = len(token_ids)
    if n == 0:
        return torch.empty(0, dtype=torch.float32)
    if step_logits is None or len(step_logits) == 0:
        raise ValueError("step_logits is empty; generate with output_logits=True.")
    if len(step_logits) < n:
        raise ValueError(
            f"Got {len(step_logits)} logit steps but {n} token ids; "
            "output may be truncated."
        )

    device = step_logits[0].device
    probs_out = []
    for i in range(n):
        logits_i = step_logits[i]
        if logits_i.dim() == 3:
            logits_i = logits_i[:, -1, :]
        tid = int(token_ids[i])
        prob_i = F.softmax(logits_i.float(), dim=-1)[0, tid]
        probs_out.append(prob_i)
    return torch.stack(probs_out).to(device)
