"""Read frozen token streams as fixed-length windows, in a fixed order.

The training stream is cut into non-overlapping windows of seq_len+1 tokens. Window w
covers tokens [w*T, w*T + T + 1). The order in which windows are visited is a function of
`data_seed` alone, never of the model seed, so every run sees the identical token stream in
the identical order. Optimizer step s consumes windows [s*W, (s+1)*W) of that order, where
W = tokens_per_step / seq_len; micro-batching only partitions those W windows. The data
position is therefore a pure function of the step, which makes resumption exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..tokenizer.bpe import Tokenizer
from .prepare import verify_manifest


class TokenData:
    def __init__(self, data_dir: str | Path, verify: bool = True, verify_splits: list[str] | None = None):
        self.dir = Path(data_dir)
        self.manifest = (
            verify_manifest(self.dir, verify_splits)
            if verify
            else json.loads((self.dir / "MANIFEST.json").read_text())
        )
        self.tokenizer = Tokenizer.load(self.dir / "tokenizer.json")
        self.vocab_size = self.manifest["tokenizer"]["vocab_size"]
        self.bos_id = self.manifest["tokenizer"]["bos_id"]
        self._splits: dict[str, np.memmap] = {}
        self.token_bytes = torch.tensor(self.tokenizer.token_byte_lengths(), dtype=torch.int64)

    @property
    def hash(self) -> str:
        """Identity of the data: the train split's sha256 plus the tokenizer hash."""
        return f"{self.manifest['tokenizer']['hash']}:{self.manifest['splits']['train']['sha256'][:16]}"

    def split(self, name: str) -> np.memmap:
        if name not in self._splits:
            info = self.manifest["splits"][name]
            self._splits[name] = np.memmap(self.dir / info["file"], dtype=np.uint16, mode="r")
        return self._splits[name]

    def has_split(self, name: str) -> bool:
        return name in self.manifest["splits"] and self.manifest["splits"][name]["n_tokens"] > 0


class WindowStream:
    def __init__(self, tokens: np.ndarray, seq_len: int, order: str = "shuffled", data_seed: int = 0):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_windows = (len(tokens) - 1) // seq_len
        if self.n_windows <= 0:
            raise ValueError(f"stream of {len(tokens)} tokens is shorter than one window of {seq_len}")
        if order == "shuffled":
            self.order = np.random.default_rng(data_seed).permutation(self.n_windows)
        elif order == "sequential":
            self.order = np.arange(self.n_windows)
        else:
            raise ValueError(f"unknown data order {order!r}")

    def windows(self, start: int, count: int) -> np.ndarray:
        """Token windows for positions [start, start+count) of the visiting order."""
        if start + count > self.n_windows:
            raise IndexError(
                f"requested window {start + count} of {self.n_windows}: the run would repeat data. "
                "Tokenize more data or train for fewer tokens."
            )
        T = self.seq_len
        idx = self.order[start : start + count]
        return np.stack([self.tokens[w * T : w * T + T + 1] for w in idx]).astype(np.int64)

    def batch(self, start: int, count: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        arr = torch.from_numpy(self.windows(start, count))
        if str(device).startswith("cuda"):
            arr = arr.pin_memory().to(device, non_blocking=True)
        else:
            arr = arr.to(device)
        return arr[:, :-1], arr[:, 1:]
