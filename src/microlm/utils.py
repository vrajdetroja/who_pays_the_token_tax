"""Small shared helpers: hashing, provenance, devices."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace. The basis of every hash we take."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_obj(obj: Any, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:length]


def sha256_file(path: str | os.PathLike, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    """Current commit, suffixed with -dirty if the tree has uncommitted changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bounded_imap(pool, fn, iterable, window: int = 16384, chunksize: int = 64):
    """Ordered `pool.imap` that reads at most `window` items ahead of the consumer.

    Plain `Pool.imap` drains its input iterator as fast as it can, which would pull an
    entire multi-GB corpus into memory before the first result is consumed.
    """
    import itertools

    it = iter(iterable)
    while batch := list(itertools.islice(it, window)):
        yield from pool.imap(fn, batch, chunksize=chunksize)
