"""Bits per byte under the two context protocols, plus the evaluation controls.

Every target token of an evaluation stream is scored exactly once, with sliding windows of
`window` tokens advanced by `stride`: the first window scores all its targets, every later
window scores only its last `stride` targets, so each scored token sees between
window - stride and window - 1 tokens of context. Because every target is scored once, the
denominator of bits per byte is the stream's full UTF-8 byte count at every dose level and
under both protocols: the scored byte span is identical by construction.

    ctx = tok   window = 512 tokens          (a deployed system's fixed token window)
    ctx = byte  window = round(512 * r)      (the same bytes as the reference's window)

Controls, run on the same windows:
    shuffle  each window's unscored prefix (its left context) is replaced by tokens from a
             distant part of the stream (positive control 2: BPB must rise)
    stride   the main protocol at a different stride (negative control 2: BPB must agree)

`wasted_mass` is the mean probability the model puts on vocabulary entries the level's
tokenizer can never produce; it is reported because those rows stay in the softmax.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EvalResult:
    nats: float
    n_tokens: int
    n_bytes: int
    wasted_mass: float | None

    @property
    def bpb(self) -> float:
        return self.nats / self.n_bytes / math.log(2)

    def bits_per(self, n_units: int) -> float:
        return self.nats / math.log(2) / n_units


def windows(n: int, window: int, stride: int) -> list[tuple[int, int, int]]:
    """(begin, end, first_scored) over input positions of a stream of n tokens.

    The window feeds inputs tokens[begin:end] and predicts tokens[begin+1:end+1]; targets at
    stream positions >= first_scored are scored. Target positions 1..n-1 are each scored once.
    """
    if not 0 < stride <= window:
        raise ValueError(f"need 0 < stride <= window, got stride {stride}, window {window}")
    out, prev_end, begin = [], 1, 0
    while prev_end < n:
        end = min(begin + window, n - 1)
        out.append((begin, end, prev_end))
        prev_end = end + 1
        begin += stride
        if end == n - 1:
            break
    return out


def _donor_starts(n: int, spans: list[tuple[int, int]], min_distance: int, seed: int) -> list[int]:
    """For each (begin, length), a start for a same-length donor span far from it."""
    rng = np.random.default_rng(seed)
    starts = []
    for begin, length in spans:
        for _ in range(1000):
            d = int(rng.integers(0, max(1, n - length)))
            if abs(d - begin) >= min_distance:
                break
        starts.append(d)
    return starts


@torch.no_grad()
def evaluate_stream(
    model,
    tokens: np.ndarray,
    token_bytes: torch.Tensor,
    window: int,
    stride: int,
    device: str,
    batch_size: int = 16,
    dtype: str = "bf16",
    shuffle_seed: int | None = None,
    unreachable: torch.Tensor | None = None,
) -> EvalResult:
    """Score every target of `tokens` once (see module docstring)."""
    n = len(tokens)
    plan = windows(n, window, stride)
    donors = None
    if shuffle_seed is not None:
        spans = [(b, f - 1 - b) for b, _, f in plan]  # unscored prefix: inputs [b, first_scored - 1)
        donors = _donor_starts(n, spans, min_distance=8 * window, seed=shuffle_seed)
    use_bf16 = dtype == "bf16" and not str(device).startswith("cpu")
    autocast = torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16) if use_bf16 else nullcontext()
    token_bytes = token_bytes.to(device)
    if unreachable is not None:
        unreachable = unreachable.to(device)
    nats, n_scored, n_bytes, wasted = 0.0, 0, 0, 0.0
    was_training = model.training
    model.eval()
    # Group windows by length so each batch is rectangular; only the last window can be short.
    for i in range(0, len(plan), batch_size):
        chunk = plan[i : i + batch_size]
        by_len: dict[int, list[int]] = {}
        for j, (b, e, _) in enumerate(chunk):
            by_len.setdefault(e - b, []).append(j)
        for length, idx in by_len.items():
            xs, ys, masks = [], [], []
            for j in idx:
                b, e, first = chunk[j]
                seg = np.asarray(tokens[b : e + 1], dtype=np.int64)
                if donors is not None:
                    k = first - 1 - b  # tokens of left context to replace
                    if k > 0:
                        d = donors[i + j]
                        seg = seg.copy()
                        seg[:k] = np.asarray(tokens[d : d + k], dtype=np.int64)
                xs.append(seg[:-1])
                ys.append(seg[1:])
                m = np.zeros(length, dtype=bool)
                m[first - 1 - b :] = True
                masks.append(m)
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys)).to(device)
            mask = torch.from_numpy(np.stack(masks)).to(device)
            with autocast:
                logits, loss = model(x, y, reduction="none")
            loss = loss.float()
            nats += loss[mask].sum().item()
            n_scored += int(mask.sum().item())
            n_bytes += int(token_bytes[y[mask]].sum().item())
            if unreachable is not None and len(unreachable):
                lf = logits.float()
                lse_all = torch.logsumexp(lf, dim=-1)
                lse_un = torch.logsumexp(lf.index_select(-1, unreachable), dim=-1)
                wasted += torch.exp(lse_un - lse_all)[mask].sum().item()
    model.train(was_training)
    if n_scored != n - 1:
        raise AssertionError(f"scored {n_scored} targets of {n - 1}")
    return EvalResult(nats, n_scored, n_bytes, wasted / n_scored if unreachable is not None else None)
