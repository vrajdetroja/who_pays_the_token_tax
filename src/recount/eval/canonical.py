"""The canonical-tokenization gap (Chirkova et al., 2023): PREREGISTRATION §7's diagnostic.

Bits per byte score one token sequence per text, the canonical one its tokenizer produces.
The text's probability under a model is the sum over every token sequence that spells it,
so canonical scoring overstates BPB by log2(p(text) / p(canonical)) / bytes >= 0. If that gap
grew with the dose, part of the measured penalty would be an artifact of how it is scored.

p(text) is estimated by importance sampling, with the proposal Chirkova et al. use: token
sequences drawn left to right from the model itself, restricted at each step to the tokens
that spell the text's next bytes and that the level can produce (single bytes always can, so
no draw dead-ends). For a draw t with Z_i the probability the model puts on the allowed set at
step i, p(t) / q(t) = prod_i Z_i. The canonical sequence's own probability is computed
exactly and kept out of the sampled sum:

    p_hat(text) = p(canonical) + mean_k [ prod_i Z_i^(k) * 1(t_k != canonical) ]

which is unbiased, never below p(canonical), and lower-variance than plain sampling when the
canonical sequence carries most of the mass. log p_hat is biased low (Jensen), so the gap is,
if anything, understated; the half-sample estimate is reported beside it as a check.

The sentences are FLORES-200 devtest, recovered from each shard's flores split (articles of
newline-joined sentences), so they are exactly the text the models were evaluated on. Each
is scored alone after BOS.

    python -m recount.eval.canonical --model M --seed 0 --out results/canonical_gap.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from microlm.train.checkpoint import load_checkpoint
from microlm.utils import pick_device

from ..model import DoseLM
from ..model import build as build_model
from ..study import load_study
from ..tok.dose import DoseTokenizer
from ..tok.shard import ShardData
from ..train import plan
from ..train.trainer import find_run, masking


def flores_sentences(data: ShardData) -> list[bytes]:
    """FLORES devtest sentences as the shard holds them: documents (BOS + ids) are articles,
    their sentences joined by newlines."""
    toks = np.asarray(data.split("flores"))
    vb = data.tokenizer.vocab_bytes
    starts = list(np.flatnonzero(toks == data.bos_id)) + [len(toks)]
    out: list[bytes] = []
    for a, b in zip(starts[:-1], starts[1:]):
        out += b"".join(vb[i] for i in toks[a + 1 : b]).split(b"\n")
    n = data.manifest["splits"]["flores"].get("n_sentences")
    if n is not None and len(out) != n:
        raise ValueError(f"recovered {len(out)} sentences, the manifest says {n}")
    return out


def reachable_trie(tok: DoseTokenizer) -> dict:
    """Byte trie over the tokens the level can produce (not BOS); a node's -1 key is its id."""
    root: dict = {}
    for i in range(256 + tok.active):
        node = root
        for byte in tok.vocab_bytes[i]:
            node = node.setdefault(byte, {})
        node[-1] = i
    return root


def allowed_rows(trie: dict, text: bytes) -> list[list[tuple[int, int]]]:
    """For each byte offset j: (id, length) of every reachable token spelling text[j:j+length]."""
    rows = []
    for j in range(len(text)):
        node, hits = trie, []
        for k in range(j, len(text)):
            node = node.get(text[k])
            if node is None:
                break
            if -1 in node:
                hits.append((node[-1], k + 1 - j))
        rows.append(hits)
    return rows


def allowed_tables(trie: dict, texts: bytes | list[bytes], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """allowed_rows of each text, stacked text after text and padded to the widest offset:
    token ids, their lengths, and a validity mask."""
    rows = [r for t in ([texts] if isinstance(texts, bytes) else texts) for r in allowed_rows(trie, t)]
    width = max(len(h) for h in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    lens = torch.zeros(len(rows), width, dtype=torch.long)
    valid = torch.zeros(len(rows), width, dtype=torch.bool)
    for j, hits in enumerate(rows):
        for c, (i, n) in enumerate(hits):
            ids[j, c], lens[j, c], valid[j, c] = i, n, True
    return ids.to(device), lens.to(device), valid.to(device)


def last_logits(lm: DoseLM, idx: torch.Tensor) -> torch.Tensor:
    """Masked next-token logits at the last position only: the full forward would project
    every position onto the vocabulary, which sampling never needs."""
    base, cfg = lm.base, lm.base.cfg
    if base.pos_emb is not None or cfg.pos_type == "sinusoidal":
        logits = base(idx)[0][:, -1]
    else:
        x = base.tok_emb(idx)
        ctx = base._context(idx.shape[1], idx.device, False, None)
        for block in base.blocks:
            x = block(x, ctx)
        x = base.final_norm(x[:, -1])
        w = base.tok_emb.weight if base.head is None else base.head.weight
        logits = F.linear(x, w[: cfg.vocab_size])
    logits = logits.float()
    return logits.masked_fill(lm.unreachable, float("-inf")) if lm.masked else logits


@torch.no_grad()
def sequence_nats(lm: DoseLM, bos: int, ids: list[int], device) -> float:
    """-log p(ids | BOS), teacher-forced."""
    seq = torch.tensor([[bos] + ids], dtype=torch.long, device=device)
    _, loss = lm(seq[:, :-1], seq[:, 1:], reduction="none")
    return float(loss.float().sum())


@torch.no_grad()
def sample_weights(lm: DoseLM, bos: int, texts: list[bytes], trie: dict, canonicals: list[list[int]], k: int,
                   generator: torch.Generator, device, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """k proposal draws per text, every text at once: each draw adds one token per step, so
    the unfinished draws always form a rectangular batch. Returns (texts, k) arrays: each
    draw's log prod_i Z_i, and whether it is its text's canonical sequence."""
    ids_t, lens_t, valid_t = allowed_tables(trie, texts, device)
    n_bytes = torch.tensor([len(t) for t in texts], device=device)
    first_row = torch.cumsum(n_bytes, 0) - n_bytes
    text_of = torch.arange(len(texts), device=device).repeat_interleave(k)
    end, pos = n_bytes[text_of], torch.zeros_like(text_of)
    logw = torch.zeros(len(text_of), dtype=torch.float32, device=device)  # MPS has no float64
    seqs = torch.full((len(text_of), int(n_bytes.max()) + 1), bos, dtype=torch.long, device=device)
    active, step = torch.arange(len(text_of), device=device), 0
    while len(active):
        step += 1
        for part in active.split(chunk):
            probs = torch.softmax(last_logits(lm, seqs[part, :step]), dim=-1)
            row = first_row[text_of[part]] + pos[part]
            cand, lens = ids_t[row], lens_t[row]
            p = probs.gather(1, cand) * valid_t[row]
            z = p.sum(1)
            logw[part] += torch.log(z)
            pick = torch.multinomial(p / z[:, None], 1, generator=generator).squeeze(1)
            r = torch.arange(len(part), device=device)
            seqs[part, step] = cand[r, pick]
            pos[part] += lens[r, pick]
        active = active[pos[active] < end[active]]
    # A draw that starts with its text's whole canonical sequence has spelled the text, so it stopped there.
    width = max(map(len, canonicals))
    canon = torch.full((len(texts), width), -1, dtype=torch.long)
    for i, c in enumerate(canonicals):
        canon[i, : len(c)] = torch.tensor(c)
    canon = canon.to(device)[text_of]
    is_canon = ((seqs[:, 1 : width + 1] == canon) | (canon < 0)).all(1)
    shape = (len(texts), k)
    return logw.cpu().double().numpy().reshape(shape), is_canon.cpu().numpy().reshape(shape)


def marginal_nats(canon_nats: float, logw: np.ndarray, is_canon: np.ndarray) -> float:
    """-log p_hat, with p_hat = p(canonical) + mean over draws of the non-canonical weights."""
    others = np.where(is_canon, -np.inf, logw)
    mean_others = np.logaddexp.reduce(others) - math.log(len(logw)) if np.isfinite(others).any() else -np.inf
    return -float(np.logaddexp(-canon_nats, mean_others))


def gap_rows(lm: DoseLM, tok: DoseTokenizer, sentences: list[bytes], k: int, seed: int, device,
             chunk: int = 2048) -> list[dict]:
    canonicals = [tok.encode_fast(t.decode("utf-8")) for t in sentences]
    for t, c in zip(sentences, canonicals):
        assert b"".join(tok.vocab_bytes[i] for i in c) == t
    canon_nats = [sequence_nats(lm, tok.bos_id, c, device) for c in canonicals]
    gen = torch.Generator(device=device).manual_seed(seed)
    logw, is_canon = sample_weights(lm, tok.bos_id, sentences, reachable_trie(tok), canonicals, k, gen, device, chunk)
    return [{"sentence": i, "n_bytes": len(t), "canonical_tokens": len(c), "canonical_nats": cn,
             "marginal_nats": marginal_nats(cn, w, ic), "marginal_nats_half": marginal_nats(cn, w[: k // 2], ic[: k // 2]),
             "draws_canonical": float(ic.mean())}
            for i, (t, c, cn, w, ic) in enumerate(zip(sentences, canonicals, canon_nats, logw, is_canon))]


def append_csv(path: Path, rows: list[dict]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def pick(n: int, size: int, seed: int) -> list[int]:
    return sorted(np.random.default_rng(seed).choice(n, size=min(size, n), replace=False).tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="queues/main.yaml")
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="M")
    ap.add_argument("--seed", type=int, default=0, help="the training seed whose endpoints are scored")
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--endpoints", nargs="*", default=["final", "deployed"])
    ap.add_argument("--sentences", type=int, default=200)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2048, help="draws per forward pass")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--untrained", action="store_true", help="dry run: fresh weights, no checkpoint")
    ap.add_argument("--out", default="results/canonical_gap.csv")
    args = ap.parse_args()

    study, root, device = load_study(args.study), Path(args.root), pick_device(args.device)
    jobs = [j for j in plan.load(Path(args.queue)) if j.model == args.model and j.seed == args.seed
            and j.tokenizer == "main" and j.endpoint in args.endpoints and (args.langs is None or j.lang in args.langs)]
    out = Path(args.out)
    for job in sorted(jobs, key=lambda j: (j.lang, j.level, j.endpoint)):
        t0 = time.time()
        data = ShardData(root / job.shard_dir, verify_splits=["flores"])
        sents = flores_sentences(data)
        chosen = pick(len(sents), args.sentences, args.sample_seed)
        run = job.run_config(study)
        lm = build_model(run.model, data.tokenizer.unreachable_ids(), masking(study)).to(device).eval()
        if not args.untrained:
            ckpt = load_checkpoint(find_run(root, job.name) / "model_final.pt", map_location=device)
            lm.load_state_dict(ckpt["model"])
        rows = gap_rows(lm, data.tokenizer, [sents[i] for i in chosen], args.draws, args.sample_seed, device,
                        args.chunk)
        common = {"lang": job.lang, "model": job.model, "seed": job.seed, "level": job.level, "endpoint": job.endpoint,
                  "draws": args.draws, "untrained": args.untrained}
        append_csv(out, [common | r | {"sentence": chosen[r["sentence"]]} for r in rows])
        b = sum(r["n_bytes"] for r in rows)
        gap = sum(r["canonical_nats"] - r["marginal_nats"] for r in rows) / b / math.log(2)
        print(f"{job.name}: gap {gap:.5f} bits/byte over {len(rows)} sentences, {time.time() - t0:.0f}s", flush=True)
        done += 1


if __name__ == "__main__":
    main()
