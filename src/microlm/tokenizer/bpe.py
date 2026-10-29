"""Byte-level BPE, written from scratch.

Vocabulary layout for a tokenizer with M merges:
    ids 0..255          raw bytes
    ids 256..256+M-1    merges, in the order they were learned
    id  256+M           <|bos|>, the document separator

so vocab_size = 257 + M. Because BPE training is greedy and its merge sequence does not
depend on the target size, the first k merges of a 32k tokenizer *are* the k-merge
tokenizer. One training run therefore yields every vocab size in the ablation
(see `Tokenizer.truncated`), and `tests/test_tokenizer.py` checks that property.

Ties between equally frequent pairs are broken by the smaller (left, right) id pair, in
both trainers, so the fast trainer can be checked against the naive one exactly.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import regex

from ..utils import canonical_json, hash_obj

# The GPT-4 (cl100k) pre-tokenization pattern. Every character matches some branch, so
# the chunks always concatenate back to the input.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)
BOS_TOKEN = "<|bos|>"
FORMAT = "microlm-bpe-v1"
Pair = tuple[int, int]


def merge_ids(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair`, scanning left to right."""
    a, b = pair
    out: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# ---------------------------------------------------------------------------------------
# Pre-tokenization


def count_chunks(texts: Iterable[str], pattern: str = GPT4_SPLIT_PATTERN) -> Counter[bytes]:
    """Split each text with the regex and count the resulting byte chunks."""
    compiled = regex.compile(pattern)
    counts: Counter[bytes] = Counter()
    for text in texts:
        counts.update(c.encode("utf-8", "surrogatepass") for c in compiled.findall(text))
    return counts


def _count_worker(texts: list[str]) -> Counter[bytes]:
    return count_chunks(texts)


def count_chunks_parallel(texts: Iterable[str], workers: int, batch: int = 512) -> Counter[bytes]:
    """`count_chunks` over a process pool; the result does not depend on `workers`."""
    if workers <= 1:
        return count_chunks(texts)
    import multiprocessing as mp

    def batches():
        buf: list[str] = []
        for t in texts:
            buf.append(t)
            if len(buf) == batch:
                yield buf
                buf = []
        if buf:
            yield buf

    from ..utils import bounded_imap

    total: Counter[bytes] = Counter()
    with mp.get_context("spawn").Pool(workers) as pool:
        for c in bounded_imap(pool, _count_worker, batches(), window=8 * workers, chunksize=1):
            total.update(c)
    return total


# ---------------------------------------------------------------------------------------
# Trainers


def _best_pair(pair_counts: Mapping[Pair, int]) -> Pair | None:
    best = None
    for p, c in pair_counts.items():
        if c > 0 and (best is None or (-c, p) < best[0]):
            best = ((-c, p), p)
    return None if best is None else best[1]


def train_bpe_naive(chunk_counts: Mapping[bytes, int], num_merges: int) -> list[Pair]:
    """Reference trainer: recount every pair after every merge. O(corpus) per merge.

    Kept only to check `train_bpe` against and to measure the speedup.
    """
    words = [list(chunk) for chunk in chunk_counts]
    freqs = list(chunk_counts.values())
    merges: list[Pair] = []
    for m in range(num_merges):
        pair_counts: dict[Pair, int] = defaultdict(int)
        for w, f in zip(words, freqs):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
        best = _best_pair(pair_counts)
        if best is None:
            break
        merges.append(best)
        words = [merge_ids(w, best, 256 + m) for w in words]
    return merges


def train_bpe(
    chunk_counts: Mapping[bytes, int], num_merges: int, progress: bool = False
) -> list[Pair]:
    """Fast trainer: incremental pair counts plus a max-heap with lazy invalidation.

    After a merge, only the words that contained the merged pair are touched, and the pair
    counts are updated by the difference between each word's old and new pairs. A heap entry
    is valid only if its count still equals the live count; stale entries are skipped when
    popped. Each count change pushes a fresh entry, so the live count of every pair always
    has an entry, and the first valid pop is exactly the naive trainer's argmax.
    """
    words = [list(chunk) for chunk in chunk_counts]
    freqs = list(chunk_counts.values())
    pair_counts: dict[Pair, int] = defaultdict(int)
    where: dict[Pair, set[int]] = defaultdict(set)  # pair -> words that (may) contain it
    for wi, (w, f) in enumerate(zip(words, freqs)):
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            where[p].add(wi)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=num_merges, desc="bpe merges", unit="merge")

    merges: list[Pair] = []
    for m in range(num_merges):
        best = None
        while heap:
            neg_c, p = heapq.heappop(heap)
            if -neg_c > 0 and pair_counts.get(p, 0) == -neg_c:
                best = p
                break
        if best is None:
            break  # every word is a single token
        new_id = 256 + m
        merges.append(best)

        delta: dict[Pair, int] = defaultdict(int)
        for wi in where.pop(best):
            w = words[wi]
            new_w = merge_ids(w, best, new_id)
            if len(new_w) == len(w):
                continue  # stale index entry
            f = freqs[wi]
            for p in zip(w, w[1:]):
                delta[p] -= f
            for p in zip(new_w, new_w[1:]):
                delta[p] += f
                where[p].add(wi)
            words[wi] = new_w
        for p, d in delta.items():
            if d == 0:
                continue
            c = pair_counts[p] + d
            if c > 0:
                pair_counts[p] = c
                heapq.heappush(heap, (-c, p))
            else:
                del pair_counts[p]
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return merges


# ---------------------------------------------------------------------------------------
# Tokenizer


class Tokenizer:
    def __init__(self, merges: list[Pair], pattern: str = GPT4_SPLIT_PATTERN):
        self.merges: list[Pair] = [tuple(p) for p in merges]
        self.pattern = pattern
        self._re = regex.compile(pattern)
        self.ranks: dict[Pair, int] = {p: i for i, p in enumerate(self.merges)}
        self.vocab_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab_bytes.append(self.vocab_bytes[a] + self.vocab_bytes[b])
        self.bos_id = 256 + len(self.merges)
        self.vocab_bytes.append(b"")  # BOS decodes to nothing and covers zero bytes
        self.vocab_size = len(self.vocab_bytes)
        self._cache: dict[bytes, list[int]] = {}

    # -- encoding ---------------------------------------------------------------------

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk)
        ranks = self.ranks
        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, p
            if best_pair is None:
                break
            ids = merge_ids(ids, best_pair, 256 + best_rank)
        if len(self._cache) > 1_000_000:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def encode_bytes(self, data: bytes) -> list[int]:
        """Encode arbitrary bytes, including invalid UTF-8. decode_bytes inverts it exactly."""
        text = data.decode("utf-8", "surrogateescape")
        out: list[int] = []
        for chunk in self._re.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8", "surrogateescape")))
        return out

    def encode(self, text: str) -> list[int]:
        """Encode a str. Lone surrogates are allowed and survive the round trip."""
        try:
            chunks = [c.encode("utf-8") for c in self._re.findall(text)]
        except UnicodeEncodeError:
            return self.encode_bytes(text.encode("utf-8", "surrogatepass"))
        out: list[int] = []
        for chunk in chunks:
            out.extend(self._encode_chunk(chunk))
        return out

    # -- decoding ---------------------------------------------------------------------

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        vb = self.vocab_bytes
        return b"".join(vb[i] for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        data = self.decode_bytes(ids)
        try:
            return data.decode("utf-8", "surrogatepass")
        except UnicodeDecodeError:  # e.g. a sequence cut mid-character
            return data.decode("utf-8", "replace")

    def token_byte_lengths(self) -> list[int]:
        """Bytes covered by each id; the denominator of bits-per-byte."""
        return [len(b) for b in self.vocab_bytes]

    # -- persistence ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "pattern": self.pattern,
            "merges": [list(p) for p in self.merges],
            "special_tokens": {BOS_TOKEN: self.bos_id},
        }

    @property
    def hash(self) -> str:
        """Content hash of the tokenizer. Recorded in every data manifest."""
        return hash_obj(self.to_dict(), length=16)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(canonical_json(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        d = json.loads(Path(path).read_text())
        if d.get("format") != FORMAT:
            raise ValueError(f"{path}: unknown tokenizer format {d.get('format')!r}")
        tok = cls(d["merges"], d["pattern"])
        if d["special_tokens"].get(BOS_TOKEN) != tok.bos_id:
            raise ValueError(f"{path}: BOS id does not match merge count")
        return tok

    def truncated(self, vocab_size: int) -> "Tokenizer":
        """The tokenizer with `vocab_size - 257` merges: a prefix of this one's merges."""
        n = vocab_size - 257
        if not 0 <= n <= len(self.merges):
            raise ValueError(f"vocab_size {vocab_size} not in [257, {self.vocab_size}]")
        return Tokenizer(self.merges[:n], self.pattern)


# ---------------------------------------------------------------------------------------
# Parallel batch encoding (used by the data pipeline)

_WORKER_TOK: Tokenizer | None = None


def _init_encode_worker(path: str) -> None:
    global _WORKER_TOK
    _WORKER_TOK = Tokenizer.load(path)


def _encode_worker(text: str) -> list[int]:
    assert _WORKER_TOK is not None
    return _WORKER_TOK.encode(text)
