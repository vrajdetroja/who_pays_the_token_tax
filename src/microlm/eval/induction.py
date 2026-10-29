"""Synthetic probes: repeated random tokens (Olsson et al., 2022).

A sequence of n random tokens is shown twice. On the second copy a model that has
induction heads can predict every token by finding the previous occurrence of the current
token and copying what followed it.

  prefix-matching score  per head: mean attention from position n+i (second copy) to
                         position i+1 (the token after the earlier occurrence)
  copy accuracy          argmax accuracy on the second copy
  in-context gain        mean loss on the first copy minus mean loss on the second
"""

from __future__ import annotations

import torch


@torch.no_grad()
def induction_probe(model, vocab_size: int, bos_id: int, device: str, n: int = 256, batch: int = 8, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    rand = torch.randint(0, vocab_size - 1, (batch, n), generator=g)
    rand[rand >= bos_id] += 1  # never sample BOS
    bos = torch.full((batch, 1), bos_id)
    seq = torch.cat([bos, rand, rand], dim=1).to(device)  # length 2n+1
    x, y = seq[:, :-1], seq[:, 1:]

    scores: list[torch.Tensor] = []
    # x = [BOS, r_1..r_n, r_1..r_{n-1}]: x index n+i holds the second r_i (1 <= i < n), whose
    # first occurrence is at index i, so an induction head attends to index i+1.
    q_pos = torch.arange(n + 1, 2 * n)
    k_pos = q_pos - n + 1

    def hook(probs: torch.Tensor) -> None:  # (B, H, T, T)
        scores.append(probs[:, :, q_pos, k_pos].mean(dim=(0, 2)).cpu())

    was_training = model.training
    model.eval()
    logits, loss = model(x, y, reduction="none", probs_hook=hook)
    model.train(was_training)
    # Target y[n] = r_1 follows r_n and is not predictable; the inducible targets start at n+1.
    first, second = loss[:, :n].mean().item(), loss[:, n + 1 :].mean().item()
    acc = (logits[:, n + 1 :].argmax(-1) == y[:, n + 1 :]).float().mean().item()
    per_head = torch.stack(scores)  # (L, H)
    return {
        "prefix_matching": per_head.tolist(),
        "max_prefix_matching": per_head.max().item(),
        "copy_accuracy": acc,
        "in_context_gain": first - second,
    }
