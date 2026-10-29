"""Loss-based evaluation: val loss, bits per byte, loss by position, length extrapolation.

Bits per byte is computed exactly: total nats over the evaluated targets divided by the
total UTF-8 bytes those target tokens decode to, divided by ln 2. Per-token cross-entropy
is not comparable across tokenizers (a bigger vocab predicts fewer, harder tokens); BPB is.
BOS targets cover zero bytes but their loss is counted, identically for every tokenizer,
since every tokenizer marks each document with exactly one BOS.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
import torch


def _autocast(device: str, dtype: str):
    if dtype == "bf16" and device != "cpu":
        return torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate_tokens(
    model,
    tokens: np.ndarray,
    token_bytes: torch.Tensor,
    seq_len: int,
    max_tokens: int | None,
    batch_size: int,
    device: str,
    dtype: str = "bf16",
    position_loss: bool = False,
) -> dict:
    """Non-overlapping windows from the start of `tokens`, in order (deterministic).

    Returns loss (nats/token), bpb, the counts behind them, and optionally the mean loss at
    each position in the window.
    """
    n_windows = (len(tokens) - 1) // seq_len
    if max_tokens is not None:
        n_windows = min(n_windows, max(1, max_tokens // seq_len))
    # Per-batch reductions run on-device in fp32; totals accumulate on the CPU in fp64
    # (MPS has no fp64, and 10M-token sums deserve it).
    token_bytes = token_bytes.to(device)
    nats, n_bytes = 0.0, 0
    pos_nats = torch.zeros(seq_len, dtype=torch.float64)
    was_training = model.training
    model.eval()
    for start in range(0, n_windows, batch_size):
        ws = range(start, min(start + batch_size, n_windows))
        arr = np.stack([tokens[w * seq_len : w * seq_len + seq_len + 1] for w in ws]).astype(np.int64)
        batch = torch.from_numpy(arr).to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        with _autocast(device, dtype):
            _, loss = model(x, y, reduction="none")
        loss = loss.float()
        nats += loss.sum().item()
        n_bytes += int(token_bytes[y].sum().item())
        if position_loss:
            pos_nats += loss.sum(0).cpu().double()
    model.train(was_training)
    n_tok = n_windows * seq_len
    out = {
        "loss": nats / n_tok,
        "bpb": nats / n_bytes / math.log(2),
        "n_tokens": n_tok,
        "n_bytes": n_bytes,
    }
    if position_loss:
        out["position_loss"] = (pos_nats / n_windows).tolist()
    return out


def doc_starts(tokens: np.ndarray, bos_id: int) -> np.ndarray:
    return np.flatnonzero(tokens == bos_id)


@torch.no_grad()
def extrapolation_eval(
    model,
    tokens: np.ndarray,
    bos_id: int,
    token_bytes: torch.Tensor,
    lengths: list[int],
    device: str,
    dtype: str = "bf16",
    max_docs: int = 200,
    n_buckets: int = 64,
) -> dict:
    """Loss at evaluation context lengths beyond the training length.

    Every length is evaluated on the *same* documents (each at least max(lengths)+1 tokens
    long), reading the first L+1 tokens of each, so the curves differ only in context
    length. Also returns loss bucketed by position, which shows where a method breaks.
    """
    L_max = max(lengths)
    starts = doc_starts(tokens, bos_id)
    ends = np.append(starts[1:], len(tokens))
    usable = [s for s, e in zip(starts, ends) if e - s >= L_max + 1][:max_docs]
    if not usable:
        return {"n_docs": 0, "lengths": {}, "error": f"no documents with >= {L_max + 1} tokens"}
    token_bytes = token_bytes.to(device)
    results = {"n_docs": len(usable), "lengths": {}}
    was_training = model.training
    model.eval()
    for L in lengths:
        nats, n_bytes = 0.0, 0
        pos = torch.zeros(L, dtype=torch.float64)
        try:
            for s in usable:
                batch = torch.from_numpy(tokens[s : s + L + 1].astype(np.int64))[None].to(device)
                x, y = batch[:, :-1], batch[:, 1:]
                with _autocast(device, dtype):
                    _, loss = model(x, y, reduction="none")
                loss = loss[0].float()
                nats += loss.sum().item()
                n_bytes += int(token_bytes[y].sum().item())
                pos += loss.cpu().double()
        except ValueError as e:  # learned positions cannot extrapolate
            results["lengths"][L] = {"error": str(e)}
            continue
        pos = (pos / len(usable)).numpy()
        edges = np.linspace(0, L, n_buckets + 1).astype(int)
        results["lengths"][L] = {
            "loss": nats / (L * len(usable)),
            "bpb": nats / n_bytes / math.log(2),
            "bucket_edges": edges.tolist(),
            "bucket_loss": [float(pos[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])],
        }
    model.train(was_training)
    return results
