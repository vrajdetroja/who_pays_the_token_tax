"""Benchmark the incremental BPE trainer against the naive reference.

    python -m microlm.tokenizer.bench --input data/fineweb_edu/tok_train_sample.jsonl \
        --naive-mb 2 --naive-merges 500 --fast-mb 100 --fast-merges 8000

Both trainers run on the same chunk counts and must produce identical merges. The naive
trainer is O(corpus) per merge, so it is timed on a small slice and the fast trainer on a
large one; the speedup is reported on the slice they share.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

from .bpe import count_chunks_parallel, train_bpe, train_bpe_naive
from .train import iter_jsonl_texts


def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1e9 if sys.platform == "darwin" else 1e6)  # bytes on macOS, KiB on Linux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--naive-mb", type=float, default=2)
    ap.add_argument("--naive-merges", type=int, default=500)
    ap.add_argument("--fast-mb", type=float, default=100)
    ap.add_argument("--fast-merges", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    small = count_chunks_parallel(iter_jsonl_texts(args.input, int(args.naive_mb * 1e6)), args.workers)
    t0 = time.time()
    naive = train_bpe_naive(small, args.naive_merges)
    t_naive = time.time() - t0
    t0 = time.time()
    fast = train_bpe(small, args.naive_merges)
    t_fast = time.time() - t0
    assert fast == naive, "fast and naive trainers disagree"

    t0 = time.time()
    big = count_chunks_parallel(iter_jsonl_texts(args.input, int(args.fast_mb * 1e6)), args.workers)
    t_count = time.time() - t0
    t0 = time.time()
    merges = train_bpe(big, args.fast_merges)
    t_big = time.time() - t0

    report = {
        "slice": {"mb": args.naive_mb, "unique_chunks": len(small), "merges": args.naive_merges,
                  "naive_s": round(t_naive, 2), "fast_s": round(t_fast, 3), "speedup": round(t_naive / t_fast, 1),
                  "identical": True},
        "large": {"mb": args.fast_mb, "unique_chunks": len(big), "merges": len(merges),
                  "pretokenize_s": round(t_count, 1), "merge_s": round(t_big, 1), "peak_rss_gb": round(peak_rss_gb(), 2)},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
