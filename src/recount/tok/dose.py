"""The fertility dose: one language's full tokenizer, and prefixes of its merges.

Every dose level of a language lives in the *same* id space of 16,384 entries:

    ids 0..255         raw bytes
    ids 256..16382     the 16,127 merges of the full tokenizer, in the order learned
    id  16383          <|bos|>, the document separator (fixed at every level)

A level keeps only the first k merges active. Encoding never applies a later merge, so ids
256 + k .. 16382 are unreachable: they stay in the embedding table and the softmax as unused
rows, which keeps the parameter count identical at every level. Because BPE training is
greedy, the first k merges of the full tokenizer *are* the k-merge tokenizer (microlm's
`Tokenizer.truncated`), so a level is a vocabulary allocation, not a different algorithm.
The dose r = (tokens per byte at k) / (tokens per byte at the full merge count), measured on
held-out text.

Pre-tokenization uses GPT-4's split pattern with one change: combining marks (\\p{M}) stay
with letters, as in o200k_base. GPT-4's pattern splits Tamil (and other Indic) words at
every vowel sign, which would make even the full tokenizer artificially fertile.

Training and bulk encoding run in Hugging Face `tokenizers` (Rust): the from-scratch Python
trainer's pair index does not fit 200 MB of unspaced Chinese in 16 GB of memory, and Python
encoding is too slow for gigabytes. microlm's Python encoder remains the reference;
`tests/test_dose.py` checks the two agree token for token.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from microlm.tokenizer.bpe import Tokenizer
from microlm.utils import canonical_json, hash_obj

MARK_AWARE_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{M}\p{N}]?+[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)
FORMAT = "recount-dose-v1"


def _byte_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-character map, which HF byte-level BPE uses."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_CHAR = _byte_unicode()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


def to_hf(data: bytes) -> str:
    return "".join(BYTE_TO_CHAR[b] for b in data)


def from_hf(s: str) -> bytes:
    return bytes(CHAR_TO_BYTE[c] for c in s)


class DoseTokenizer(Tokenizer):
    """A prefix of `active` merges, inside the full tokenizer's id space."""

    def __init__(self, merges: list, active: int | None = None, pattern: str = MARK_AWARE_PATTERN, meta: dict | None = None):
        super().__init__(merges, pattern)  # full id space: bos = 256 + len(merges)
        self.active = len(self.merges) if active is None else int(active)
        if not 0 <= self.active <= len(self.merges):
            raise ValueError(f"active {self.active} not in [0, {len(self.merges)}]")
        self.ranks = {p: i for i, p in enumerate(self.merges[: self.active])}  # what encoding may apply
        self._cache = {}
        self.meta = dict(meta or {})
        self._hf = None

    def with_active(self, active: int, **meta) -> "DoseTokenizer":
        return DoseTokenizer(self.merges, active, self.pattern, self.meta | meta)

    def unreachable_ids(self) -> list[int]:
        return list(range(256 + self.active, 256 + len(self.merges)))

    # -- fast encoding via HF tokenizers ------------------------------------------------

    def hf(self):
        """An HF tokenizer that encodes exactly like `encode` (ids in this id space)."""
        if self._hf is None:
            from tokenizers import Regex, Tokenizer as HFTokenizer, decoders, models, pre_tokenizers

            vb = self.vocab_bytes
            vocab = {to_hf(vb[i]): i for i in range(256 + self.active)}
            merges = [(to_hf(vb[a]), to_hf(vb[b])) for a, b in self.merges[: self.active]]
            tok = HFTokenizer(models.BPE(vocab=vocab, merges=merges))
            tok.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ])
            tok.decoder = decoders.ByteLevel()
            self._hf = tok
        return self._hf

    def encode_fast(self, text: str) -> list[int]:
        return self.hf().encode(text, add_special_tokens=False).ids

    def encode_batch_fast(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.hf().encode_batch(texts, add_special_tokens=False)]

    def __getstate__(self):  # HF tokenizers are rebuilt in worker processes
        d = self.__dict__.copy()
        d["_hf"] = None
        return d

    # -- persistence --------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"format": FORMAT, "pattern": self.pattern, "merges": [list(p) for p in self.merges],
                "active": self.active, "special_tokens": {"<|bos|>": self.bos_id}, "meta": self.meta}

    @property
    def hash(self) -> str:
        return hash_obj({k: v for k, v in self.to_dict().items() if k != "meta"}, length=16)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(canonical_json(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "DoseTokenizer":
        d = json.loads(Path(path).read_text())
        if d.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} tokenizer")
        tok = cls(d["merges"], d["active"], d["pattern"], d.get("meta"))
        if d["special_tokens"]["<|bos|>"] != tok.bos_id:
            raise ValueError(f"{path}: BOS id does not match the merge count")
        return tok


# ---------------------------------------------------------------------------------------
# Training


def train_full(texts: Iterable[str], n_merges: int = 16_127, pattern: str = MARK_AWARE_PATTERN,
               length: int | None = None) -> DoseTokenizer:
    """Train byte-level BPE with HF's trainer, convert to microlm's (left id, right id) merges.

    HF's trainer can learn a merge whose result string already exists (the same bytes
    reached by a different split). Those duplicates are dropped, so every id has distinct
    bytes and the fast and reference encoders agree. HF is asked for a little more than
    `n_merges` so the list is still `n_merges` long after deduplication.
    """
    from tokenizers import Regex, Tokenizer as HFTokenizer, models, pre_tokenizers, trainers

    tok = HFTokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    trainer = trainers.BpeTrainer(vocab_size=256 + n_merges + 512, min_frequency=2, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    hf_merges = json.loads(tok.to_str())["model"]["merges"]
    ids = {bytes([i]): i for i in range(256)}
    merges: list[tuple[int, int]] = []
    dropped = 0
    for m in hf_merges:
        a, b = m.split(" ") if isinstance(m, str) else m
        ba, bb = from_hf(a), from_hf(b)
        if ba + bb in ids:
            dropped += 1
            continue
        if ba not in ids or bb not in ids:  # parts of a dropped duplicate's descendants
            dropped += 1
            continue
        ids[ba + bb] = 256 + len(merges)
        merges.append((ids[ba], ids[bb]))
        if len(merges) == n_merges:
            break
    if len(merges) < n_merges:
        raise ValueError(f"only {len(merges)} merges after dropping {dropped} duplicates; train on more text")
    return DoseTokenizer(merges, pattern=pattern, meta={"hf_duplicates_dropped": dropped})


# ---------------------------------------------------------------------------------------
# Dose search


def tokens_per_byte(tok: DoseTokenizer, texts: list[str]) -> float:
    n_tok = sum(len(ids) for ids in tok.encode_batch_fast(texts))
    n_bytes = sum(len(t.encode("utf-8", "surrogatepass")) for t in texts)
    return n_tok / n_bytes


def search_active(full: DoseTokenizer, texts: list[str], target_r: float) -> tuple[int, float]:
    """The merge count k whose fertility ratio r(k) is closest to `target_r`.

    r(k) is non-increasing in k: adding the lowest-priority merge can only merge more. So a
    binary search finds the boundary, and the closer of its two sides wins.
    """
    base = tokens_per_byte(full, texts)

    def r(k: int) -> float:
        return tokens_per_byte(full.with_active(k), texts) / base

    if target_r <= 1.0:
        return full.active, 1.0
    lo, hi = 0, full.active  # r(lo) >= target > r(hi) once established
    if r(lo) < target_r:
        raise ValueError(f"even byte level (k = 0) gives r = {r(lo):.3f} < {target_r}")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if r(mid) >= target_r:
            lo = mid
        else:
            hi = mid
    r_lo, r_hi = r(lo), r(hi)
    return (lo, r_lo) if abs(r_lo - target_r) <= abs(r_hi - target_r) else (hi, r_hi)
