"""The canonical-tokenization gap by beam search (PREREGISTRATION.md, amendment 4).

The importance sampler in recount.eval.canonical cannot measure the gap under the trained
models: a draw almost never follows the canonical sequence, and the draws that leave it come
back with negligible weight, so 32 draws read about zero. The beam keeps the probable
spellings instead. Moving through a text a byte at a time, the partial
spellings that end at that byte are cut to the B most probable (the canonical sequence's prefix
is always kept), and each is extended by every reachable token that spells the next bytes. Then

    p_beam(text) = p(canonical) + the summed probability of the other complete spellings reached,

a lower bound on p(text), exact once B keeps every spelling, and the gap, log2(p_beam /
p(canonical)) / bytes, is a lower bound on the true gap. p(canonical) is the teacher-forced one,
as in the sampled estimate; the beam's own copy of it is kept as a check.

One run does, in order: on 25 sentences per endpoint, the longest opening whose canonical
tokens at every level are the sentence's own first tokens and that has at most 5,000 spellings
at r = 1, with every spelling scored teacher-forced (exact) beside the beam at B = 16; the beam
at B = 16 on the 200 sentences the sampled estimate scored; and at B = 64 on the first 25 of
them. Each endpoint's rows are written when it finishes, and a rerun skips what is written.

    python -m recount.eval.canonical_beam
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from microlm.train.checkpoint import load_checkpoint
from microlm.utils import pick_device

from ..model import DoseLM
from ..model import build as build_model
from ..study import load_study
from ..tok.dose import DoseTokenizer
from ..tok.shard import ShardData
from ..train import plan
from ..train.trainer import find_run, masking
from .canonical import (allowed_rows, allowed_tables, append_csv, flores_sentences, logits_at, pick, reachable_trie,
                        sequence_nats)


class Tree:
    """Every partial spelling the beam keeps is a node whose parent is the same spelling one
    token shorter, so the pool carries node ids rather than rows of tokens."""

    def __init__(self, device, capacity: int = 1 << 16):
        self.parent = torch.empty(capacity, dtype=torch.long, device=device)
        self.token = torch.empty(capacity, dtype=torch.long, device=device)
        self.n = 0

    def add(self, parent: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        while self.n + len(parent) > len(self.parent):
            self.parent = torch.cat([self.parent, torch.empty_like(self.parent)])
            self.token = torch.cat([self.token, torch.empty_like(self.token)])
        ids = torch.arange(self.n, self.n + len(parent), device=parent.device)
        self.parent[ids] = parent
        self.token[ids] = token
        self.n += len(parent)
        return ids

    def spell(self, node: torch.Tensor, ntok: torch.Tensor, bos: int) -> torch.Tensor:
        """BOS and each node's tokens, right-padded with BOS to the longest."""
        out = torch.full((len(node), int(ntok.max()) + 1), bos, dtype=torch.long, device=node.device)
        cur, pos = node.clone(), ntok.clone()
        for _ in range(int(ntok.max())):
            m = pos > 0
            out[m, pos[m]] = self.token[cur[m]]
            cur[m], pos[m] = self.parent[cur[m]], pos[m] - 1
        return out


def top_per_key(keys: torch.Tensor, score: torch.Tensor, always: torch.Tensor, k: int) -> torch.Tensor:
    """Mask keeping, within each key, the k highest scores, and every row marked `always`."""
    by_score = torch.argsort(score, descending=True, stable=True)
    order = by_score[torch.argsort(keys[by_score], stable=True)]
    sk = keys[order]
    idx = torch.arange(len(sk), device=keys.device)
    start = torch.ones_like(sk, dtype=torch.bool)
    start[1:] = sk[1:] != sk[:-1]
    rank = idx - torch.cummax(torch.where(start, idx, torch.zeros_like(idx)), 0).values
    keep = torch.zeros_like(always)
    keep[order] = (rank < k) | always[order]
    return keep


@torch.no_grad()
def beam_mass(lm: DoseLM, bos: int, texts: list[bytes], trie: dict, canonicals: list[list[int]], beam: int,
              device, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The beam over every text at once. Returns, per text: the log of the summed probability of
    the non-canonical complete spellings it reached (-inf if none), the canonical sequence's
    log-probability as the beam accumulated it, and how many non-canonical spellings it completed."""
    fdt = torch.float32 if torch.device(device).type == "mps" else torch.float64  # MPS has no float64
    ids_t, lens_t, valid_t = allowed_tables(trie, texts, device)
    n_bytes = torch.tensor([len(t) for t in texts], device=device)
    first_row = torch.cumsum(n_bytes, 0) - n_bytes
    span = int(n_bytes.max()) + 1  # pool keys are text * span + end byte
    canon = torch.full((len(texts), span), -1, dtype=torch.long, device=device)
    for i, c in enumerate(canonicals):
        canon[i, : len(c)] = torch.tensor(c)
    tree = Tree(device)
    # the pool, one row per partial spelling: its text, end byte, tokens, log-probability,
    # whether it is the canonical sequence's prefix, and its node in the tree
    text = torch.arange(len(texts), device=device)
    end, ntok = torch.zeros_like(text), torch.zeros_like(text)
    lp = torch.zeros(len(texts), dtype=fdt, device=device)
    on_c = torch.ones(len(texts), dtype=torch.bool, device=device)
    node = tree.add(torch.full_like(text, -1), torch.full_like(text, bos))
    done = []
    for j in range(span - 1):
        now = end == j
        if not now.any():
            continue
        order = torch.argsort(ntok[now])  # similar lengths share a forward pass
        t, n, l, c, nd = (x[now][order] for x in (text, ntok, lp, on_c, node))
        text, end, ntok, lp, on_c, node = (x[~now] for x in (text, end, ntok, lp, on_c, node))
        row = first_row[t] + j
        cand, clen, cval = ids_t[row], lens_t[row], valid_t[row]
        logp = torch.empty(cand.shape, dtype=fdt, device=device)
        for a in range(0, len(t), chunk):
            b = slice(a, a + chunk)
            ids = tree.spell(nd[b], n[b], bos)
            logp[b] = torch.log_softmax(logits_at(lm, ids, n[b]), -1).gather(1, cand[b]).to(fdt)
        r, k = cval.nonzero(as_tuple=True)
        ct, ce, cl, tok = t[r], j + clen[r, k], l[r] + logp[r, k], cand[r, k]
        cc = c[r] & (tok == canon[ct, n[r]])
        fin = ce == n_bytes[ct]
        done.append((ct[fin], cl[fin], cc[fin]))
        r, ct, ce, cl, tok, cc = (x[~fin] for x in (r, ct, ce, cl, tok, cc))
        # the children join the pool, which keeps the `beam` most probable per (text, end byte)
        keep = top_per_key(torch.cat([text * span + end, ct * span + ce]), torch.cat([lp, cl]),
                           torch.cat([on_c, cc]), beam)
        old, new = keep[: len(text)], keep[len(text) :]
        text, end = torch.cat([text[old], ct[new]]), torch.cat([end[old], ce[new]])
        ntok, lp = torch.cat([ntok[old], n[r[new]] + 1]), torch.cat([lp[old], cl[new]])
        on_c, node = torch.cat([on_c[old], cc[new]]), torch.cat([node[old], tree.add(nd[r[new]], tok[new])])
    assert len(text) == 0, "every partial spelling should have been extended or completed"
    dt, dl, dc = (torch.cat(x).cpu().numpy() for x in zip(*done))
    dl = dl.astype(np.float64)
    others, canon_lp = np.full(len(texts), -np.inf), np.full(len(texts), np.nan)
    count = np.zeros(len(texts), dtype=int)
    for i in range(len(texts)):
        o, ci = dl[(dt == i) & ~dc], dl[(dt == i) & dc]
        if len(ci) != 1:
            raise RuntimeError(f"text {i}: the canonical sequence completed {len(ci)} times")
        canon_lp[i], count[i] = ci[0], len(o)
        if len(o):
            others[i] = np.logaddexp.reduce(o)
    return others, canon_lp, count


def marginal_nats(canon_nats: float, log_ratio: float) -> float:
    """-log p_beam, with p_beam = p(canonical) (1 + the other complete spellings' probability
    relative to the canonical sequence's). Take the ratio from one scorer's numbers (the beam's
    sums against the beam's own canonical log-probability), so their rounding cancels."""
    return canon_nats - float(np.log1p(np.exp(log_ratio)))


# ---------------------------------------------------------------------------------------
# Openings: every spelling, scored by code apart from the beam's


def count_spellings(rows: list[list[tuple[int, int]]]) -> int:
    """How many token sequences spell the text whose allowed_rows these are."""
    c = [0] * len(rows) + [1]
    for j in range(len(rows) - 1, -1, -1):
        c[j] = sum(c[j + n] for _, n in rows[j])
    return c[0]


def spellings(rows: list[list[tuple[int, int]]]) -> list[list[int]]:
    out, seq = [], []

    def walk(j: int) -> None:
        if j == len(rows):
            out.append(list(seq))
            return
        for i, n in rows[j]:
            seq.append(i)
            walk(j + n)
            seq.pop()

    walk(0)
    return out


@torch.no_grad()
def spelling_nats(lm: DoseLM, bos: int, seqs: list[list[int]], device, batch: int = 256) -> np.ndarray:
    """-log p(seq | BOS) for every sequence: the model's own forward, teacher-forced, in
    right-padded batches."""
    out = []
    for a in range(0, len(seqs), batch):
        part = seqs[a : a + batch]
        w = max(map(len, part)) + 1
        idx = torch.full((len(part), w), bos, dtype=torch.long)
        for r, s in enumerate(part):
            idx[r, 1 : len(s) + 1] = torch.tensor(s)
        idx = idx.to(device)
        logits, _ = lm(idx[:, :-1])
        lp = torch.log_softmax(logits.float(), -1).gather(2, idx[:, 1:, None])[..., 0]
        real = torch.arange(w - 1, device=device)[None] < torch.tensor([len(s) for s in part], device=device)[:, None]
        out.append((-torch.where(real, lp, torch.zeros_like(lp)).sum(1)).cpu().double().numpy())
    return np.concatenate(out)


def openings(sentences: list[bytes], toks: dict[float, DoseTokenizer], cap: int, min_bytes: int = 8) -> list[bytes | None]:
    """Per sentence: its longest opening (whole characters) whose canonical tokens at every level
    are the sentence's own first tokens and that has at most `cap` spellings under the level
    with the most reachable tokens; None if that opening is shorter than `min_bytes`."""
    trie = reachable_trie(max(toks.values(), key=lambda t: t.active))
    out = []
    for sent in sentences:
        s, best = sent.decode("utf-8"), None
        full = {lv: t.encode_fast(s) for lv, t in toks.items()}
        for k in range(1, len(s) + 1):
            p = s[:k].encode("utf-8")
            if count_spellings(allowed_rows(trie, p)) > cap:
                break  # every longer opening has at least as many: a last single byte extends each
            if all(full[lv][: len(c)] == c for lv, c in ((lv, t.encode_fast(s[:k])) for lv, t in toks.items())):
                best = p
        out.append(best if best is not None and len(best) >= min_bytes else None)
    return out


# ---------------------------------------------------------------------------------------
# The session


def job_rows(job, phase: str, beam: int, lm: DoseLM, tok: DoseTokenizer, sents: list[bytes], chosen: list[int],
             opens: list[bytes | None] | None, device, chunk: int) -> list[dict]:
    common = {"lang": job.lang, "model": job.model, "seed": job.seed, "level": job.level, "endpoint": job.endpoint,
              "beam": beam}
    trie = reachable_trie(tok)
    if phase == "openings":
        rows, scored = [], [(i, o) for i, o in zip(chosen, opens) if o is not None]
        canons = [tok.encode_fast(o.decode("utf-8")) for _, o in scored]
        others, canon_lp, _ = beam_mass(lm, tok.bos_id, [o for _, o in scored], trie, canons, beam, device, chunk) \
            if scored else ([], [], [])
        by_sentence = {i: (o, c, w - b) for (i, o), c, w, b in zip(scored, canons, others, canon_lp)}
        for i in chosen:
            if i not in by_sentence:
                rows.append(common | {"sentence": i, "n_bytes": 0, "spellings": 0, "canonical_tokens": 0,
                                      "canonical_nats": math.nan, "exact_nats": math.nan, "beam_nats": math.nan})
                continue
            o, c, ratio = by_sentence[i]
            seqs = spellings(allowed_rows(trie, o))
            nats = spelling_nats(lm, tok.bos_id, seqs, device)
            cn = float(nats[seqs.index(c)])  # the canonical sequence, scored with every other spelling
            rows.append(common | {"sentence": i, "n_bytes": len(o), "spellings": len(seqs), "canonical_tokens": len(c),
                                  "canonical_nats": cn, "exact_nats": -float(np.logaddexp.reduce(-nats)),
                                  "beam_nats": marginal_nats(cn, ratio)})
        return rows
    texts = [sents[i] for i in chosen]
    canons = [tok.encode_fast(t.decode("utf-8")) for t in texts]
    cn = [sequence_nats(lm, tok.bos_id, c, device) for c in canons]
    others, canon_lp, count = beam_mass(lm, tok.bos_id, texts, trie, canons, beam, device, chunk)
    return [common | {"sentence": i, "n_bytes": len(t), "canonical_tokens": len(c), "canonical_nats": n,
                      "marginal_nats": marginal_nats(n, w - b), "completed": int(k), "canonical_check": float(b + n)}
            for i, t, c, n, w, b, k in zip(chosen, texts, canons, cn, others, canon_lp, count)]


def written(path: Path, keys: list[str]) -> dict[tuple, int]:
    if not path.exists():
        return {}
    return pd.read_csv(path).groupby(keys).size().to_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="queues/main.yaml")
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="M")
    ap.add_argument("--seed", type=int, default=0, help="the training seed whose endpoints are scored")
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--endpoints", nargs="*", default=["final", "deployed"])
    ap.add_argument("--sentences", type=int, default=200, help="the sampled estimate's sentences per language")
    ap.add_argument("--sample-seed", type=int, default=0, help="its choice of them")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--check-beam", type=int, default=64)
    ap.add_argument("--check-sentences", type=int, default=25, help="the first of the chosen, for B = 64 and openings")
    ap.add_argument("--cap", type=int, default=5000, help="most spellings of an opening at the widest level")
    ap.add_argument("--chunk", type=int, default=2048, help="partial spellings per forward pass")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--untrained", action="store_true", help="dry run: fresh weights, no checkpoint")
    ap.add_argument("--out", default="results/canonical_beam.csv")
    ap.add_argument("--openings-out", default="results/canonical_openings.csv")
    args = ap.parse_args()

    study, root, device = load_study(args.study), Path(args.root), pick_device(args.device)
    jobs = sorted([j for j in plan.load(Path(args.queue)) if j.model == args.model and j.seed == args.seed
                   and j.tokenizer == "main" and j.endpoint in args.endpoints
                   and (args.langs is None or j.lang in args.langs)], key=lambda j: (j.lang, j.level, j.endpoint))
    phases = [("openings", args.beam), ("beam", args.beam), ("beam", args.check_beam)]
    out, opens_out = Path(args.out), Path(args.openings_out)
    per_lang: dict[str, tuple] = {}
    for phase, beam in phases:
        path = opens_out if phase == "openings" else out
        have = written(path, ["lang", "level", "endpoint", "beam"])
        for job in jobs:
            if job.lang not in per_lang:
                levels = {lv: ShardData(root / sd, verify_splits=["flores"])
                          for lv, sd in sorted({(j.level, j.shard_dir) for j in jobs if j.lang == job.lang})}
                sents = flores_sentences(levels[min(levels)])
                if any(flores_sentences(d) != sents for d in levels.values()):
                    raise ValueError(f"{job.lang}: the levels' FLORES splits spell different sentences")
                chosen = pick(len(sents), args.sentences, args.sample_seed)
                toks = {lv: d.tokenizer for lv, d in levels.items()}
                opens = openings([sents[i] for i in chosen[: args.check_sentences]], toks, args.cap)
                per_lang[job.lang] = (levels, sents, chosen, opens)
            levels, sents, chosen, opens = per_lang[job.lang]
            subset = chosen if (phase, beam) == ("beam", args.beam) else chosen[: args.check_sentences]
            if have.get((job.lang, job.level, job.endpoint, beam), 0) == len(subset):
                continue
            t0, data = time.time(), levels[job.level]
            lm = build_model(job.run_config(study).model, data.tokenizer.unreachable_ids(), masking(study))
            lm = lm.to(device).eval()
            if not args.untrained:
                ckpt = load_checkpoint(find_run(root, job.name) / "model_final.pt", map_location=device)
                lm.load_state_dict(ckpt["model"])
            rows = job_rows(job, phase, beam, lm, data.tokenizer, sents, subset,
                            opens if phase == "openings" else None, device, args.chunk)
            rows = [r | {"untrained": args.untrained} for r in rows]
            append_csv(path, rows)
            r = pd.DataFrame(rows).dropna(subset=["canonical_nats"])
            b = r["n_bytes"].sum()
            bpb = lambda col: (r.canonical_nats - r[col]).sum() / b / math.log(2)  # noqa: E731
            if phase == "openings":
                msg = f"exact gap {bpb('exact_nats'):.6f}, beam {bpb('beam_nats'):.6f} bits/byte over {len(r)} openings"
            else:
                msg = (f"gap {bpb('marginal_nats'):.6f} bits/byte over {len(r)} sentences, "
                       f"canonical check {r.canonical_check.abs().max():.1e} nats")
            print(f"{phase} B={beam} {job.name}: {msg}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
