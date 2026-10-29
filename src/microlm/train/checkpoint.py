"""Checkpoints that make resumption exact.

Saved: model, optimizer, step, every RNG (python / numpy / torch / cuda), the data cursor,
the run ID, and the git SHA. The LR schedule and the data order are pure functions of the
step and the config, so they need no state of their own. Writes are atomic (temp file +
rename), so an interruption mid-save never leaves a corrupt "latest" checkpoint.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, *, model, optimizer, step: int, **extra) -> None:
    path = Path(path)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,  # number of optimizer steps completed
        "rng": rng_state(),
        **extra,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def save_model_only(path: str | Path, *, model, step: int, **extra) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "step": step, **extra}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    """Load onto the CPU unless told otherwise. Resuming must load to the CPU: the saved RNG
    states have to stay CPU ByteTensors (torch.set_rng_state rejects GPU tensors), while
    load_state_dict moves model weights and optimizer state to the parameters' device anyway."""
    # Our own files; they hold numpy RNG state, which weights_only loading rejects.
    return torch.load(path, map_location=map_location, weights_only=False)
