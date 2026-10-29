"""Train the tokenizer suite.

One run to the largest vocab, then every smaller vocab is a prefix of its merges:

    python -m microlm.tokenizer.train \
        --input data/fineweb_edu/tok_train_sample.jsonl \
        --vocab-sizes 257 2048 4096 8192 16384 32768 --out tokenizers/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ..utils import canonical_json
from .bpe import Tokenizer, count_chunks_parallel, train_bpe


def iter_jsonl_texts(path: str | Path, max_bytes: int | None = None):
    seen = 0
    with open(path) as f:
        for line in f:
            text = json.loads(line)["text"]
            yield text
            seen += len(text.encode("utf-8", "surrogatepass"))
            if max_bytes is not None and seen >= max_bytes:
                return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSONL with a 'text' field per line")
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[257, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--out", default="tokenizers")
    ap.add_argument("--max-bytes", type=float, default=None, help="stop reading input after this many bytes")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--prefix", default="bpe")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    max_bytes = int(args.max_bytes) if args.max_bytes else None

    t0 = time.time()
    counts = count_chunks_parallel(iter_jsonl_texts(args.input, max_bytes), workers=args.workers)
    t_count = time.time() - t0
    n_bytes = sum(len(c) * n for c, n in counts.items())
    print(f"pre-tokenized {n_bytes / 1e6:.1f} MB into {len(counts):,} unique chunks in {t_count:.0f}s")

    max_vocab = max(args.vocab_sizes)
    t0 = time.time()
    merges = train_bpe(counts, max_vocab - 257, progress=True)
    t_train = time.time() - t0
    print(f"learned {len(merges):,} merges in {t_train:.0f}s")

    full = Tokenizer(merges)
    report = {
        "input": str(args.input),
        "train_bytes": n_bytes,
        "unique_chunks": len(counts),
        "pretokenize_seconds": round(t_count, 1),
        "merge_seconds": round(t_train, 1),
        "tokenizers": {},
    }
    for v in sorted(args.vocab_sizes):
        tok = full.truncated(v)
        path = out / f"{args.prefix}_{v}.json"
        tok.save(path)
        report["tokenizers"][str(v)] = {"path": str(path), "hash": tok.hash}
        print(f"  vocab {v:>6}: {path}  hash={tok.hash}")
    (out / f"{args.prefix}_train_report.json").write_text(canonical_json(report))


if __name__ == "__main__":
    main()
