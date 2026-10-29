"""Learning-rate schedules as pure functions of the step.

Because the multiplier depends only on (step, config), the scheduler has no state to save:
resuming at step s reproduces the schedule exactly.
"""

from __future__ import annotations

import math

from .config import TrainConfig


def lr_multiplier(step: int, cfg: TrainConfig, total_steps: int | None = None) -> float:
    """Multiplier in [0, 1] applied to every group's base LR at 0-indexed `step`."""
    total = cfg.total_steps if total_steps is None else total_steps
    warm = cfg.warmup_steps
    if step < warm:
        return (step + 1) / warm
    lo = cfg.min_lr_ratio
    if cfg.schedule == "constant":
        return 1.0
    if cfg.schedule == "wsd":
        decay_start = max(warm, total - round(cfg.decay_frac * total))
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / max(1, total - 1 - decay_start)
        return lo + (1 - lo) * (1 - min(p, 1.0))
    p = min(1.0, (step - warm) / max(1, total - 1 - warm))  # reaches 1 on the final step
    if cfg.schedule == "cosine":
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * p))
    if cfg.schedule == "linear":
        return lo + (1 - lo) * (1 - p)
    raise ValueError(cfg.schedule)
