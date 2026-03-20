"""warmup + cosine / multistep 调度器。"""

from __future__ import annotations

import math
from typing import Any, Callable


def build_lr_lambda(total_epochs: int, scheduler_cfg: dict[str, Any]) -> Callable[[int], float]:
    mode = str(scheduler_cfg.get("type", "cosine")).lower()
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
    min_lr_factor = float(scheduler_cfg.get("min_lr_factor", 0.01))
    milestones = [int(x) for x in scheduler_cfg.get("milestones", [])]
    gamma = float(scheduler_cfg.get("gamma", 0.5))

    def fn(epoch_idx: int) -> float:
        e = int(epoch_idx)
        if warmup_epochs > 0 and e < warmup_epochs:
            return max(1e-6, float(e + 1) / float(warmup_epochs))

        if mode == "multistep":
            factor = 1.0
            for m in milestones:
                if e >= m:
                    factor *= gamma
            return max(min_lr_factor, factor)

        start = warmup_epochs
        span = max(1, total_epochs - start)
        t = min(max(e - start, 0), span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / span))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    return fn
