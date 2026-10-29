"""Training windows in document order: the common-random-numbers design.

Every dose level of a (language, seed) unit reads the same documents in the same order.
Documents (each begins with BOS) are permuted by the seed, concatenated, and cut into
consecutive windows of seq_len + 1 tokens, where seq_len = round(512 r): a window holds the
same bytes at every level on average. Optimizer step s reads windows [s W, (s + 1) W).

microlm shuffles *windows* instead. Here that would break the pairing: window w holds
different text at each level, so the first 1/r of a level-r run would not be the same text
as the first 1/r of the reference. With document order, a cooldown branch at any step has
read (to within one window) the same documents at every level, which is what makes the
as-deployed branch and the data-curve branches comparable.

The data position is a pure function of the step, so resumption and branching are exact.
"""

from __future__ import annotations

import numpy as np
import torch


def doc_bounds(tokens: np.ndarray, bos_id: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(tokens == bos_id)
    if len(starts) == 0 or starts[0] != 0:
        raise ValueError("token stream must begin with BOS and mark every document with one")
    return starts, np.append(starts[1:], len(tokens))


class DocStream:
    def __init__(self, tokens: np.ndarray, bos_id: int, seq_len: int, seed: int, n_docs: int | None = None):
        starts, ends = doc_bounds(tokens, bos_id)
        if n_docs is not None:
            if n_docs > len(starts):
                raise ValueError(f"asked for {n_docs} documents; the shard holds {len(starts)}")
            starts, ends = starts[:n_docs], ends[:n_docs]
        self.order = np.random.default_rng(seed).permutation(len(starts))
        self.stream = np.concatenate([tokens[starts[i] : ends[i]] for i in self.order])
        self.seq_len = seq_len
        self.n_windows = (len(self.stream) - 1) // seq_len
        self.n_tokens = len(self.stream)

    def windows(self, start: int, count: int) -> np.ndarray:
        if start + count > self.n_windows:
            raise IndexError(f"requested window {start + count} of {self.n_windows}: the run would repeat data")
        T = self.seq_len
        a = self.stream[start * T : (start + count) * T + 1].astype(np.int64)
        return np.lib.stride_tricks.sliding_window_view(a, T + 1)[::T][:count]

    def batch(self, start: int, count: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        arr = torch.from_numpy(np.ascontiguousarray(self.windows(start, count)))
        if str(device).startswith("cuda"):
            arr = arr.pin_memory().to(device, non_blocking=True)
        else:
            arr = arr.to(device)
        return arr[:, :-1], arr[:, 1:]
