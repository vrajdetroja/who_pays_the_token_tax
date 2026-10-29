"""Frozen token shards: one directory per (language, tokenizer).

    data/shards/<lang>/<tokenizer name>/
        train.bin  heldout.bin  [flores.bin]   flat uint16 streams; each document is BOS + ids
        tokenizer.json                         the DoseTokenizer that wrote them
        MANIFEST.json                          sha256, token / document / byte counts

Training documents are written in the canonical content order (data/text/<lang>/content.jsonl),
so the first n documents of every level's shard are the same text: a model's content subset
is a document prefix, and the seed permutes it at load time (recount.train.stream).
16,384 entries fit uint16. Output directories are never overwritten.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch

from microlm.data.prepare import SplitWriter, verify_manifest  # noqa: F401  (re-exported for callers)
from microlm.utils import atomic_write_text, git_sha, sha256_file

from .dose import DoseTokenizer


def _batched(items: Iterable, n: int) -> Iterator[list]:
    buf = []
    for x in items:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def write_split(path: Path, tok: DoseTokenizer, texts: Iterable[str], batch: int = 2048) -> dict:
    """Write BOS + ids per document, plus <split>.docbytes.npy (UTF-8 bytes per document)."""
    w = SplitWriter(path, tok.bos_id)
    doc_bytes = []
    for chunk in _batched(texts, batch):
        for text, ids in zip(chunk, tok.encode_batch_fast(chunk)):
            n = len(text.encode("utf-8", "surrogatepass"))
            w.add(ids, n)
            doc_bytes.append(n)
    info = w.close()
    np.save(path.with_suffix(".docbytes.npy"), np.asarray(doc_bytes, dtype=np.int64))
    return info


def iter_jsonl(path: Path, field: str = "text") -> Iterator[str]:
    with open(path) as f:
        for line in f:
            yield json.loads(line)[field]


def build(out: Path, tok: DoseTokenizer, splits: dict[str, Path], extra: dict | None = None) -> dict:
    """Tokenize each JSONL split into out/<split>.bin and freeze the directory."""
    if out.exists():
        raise FileExistsError(f"{out} exists; frozen data is never overwritten. Delete it deliberately.")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tok.save(tmp / "tokenizer.json")
    t0 = time.time()
    info = {}
    for name, src in splits.items():
        info[name] = write_split(tmp / f"{name}.bin", tok, iter_jsonl(src))
        info[name]["source"] = {"file": str(src), "sha256": sha256_file(src)}
        if name == "flores":
            info[name]["n_sentences"] = sum(json.loads(ln)["n_sentences"] for ln in open(src))
        n = info[name]
        print(f"  {out.name:>14} {name:>8}: {n['n_tokens']:>12,} tokens {n['n_docs']:>9,} docs "
              f"{n['n_bytes'] / max(n['n_tokens'] - n['n_docs'], 1):6.3f} bytes/token", flush=True)
    manifest = {
        "tokenizer": {"hash": tok.hash, "vocab_size": tok.vocab_size, "bos_id": tok.bos_id,
                      "active": tok.active, "meta": tok.meta},
        "dtype": "uint16",
        "splits": info,
        "git_sha": git_sha(),
        "seconds": round(time.time() - t0, 1),
    } | (extra or {})
    atomic_write_text(tmp / "MANIFEST.json", json.dumps(manifest, indent=2))
    os.replace(tmp, out)
    return manifest


class ShardData:
    """A frozen shard directory: memmapped splits, the tokenizer, bytes per token id."""

    def __init__(self, path: str | Path, verify: bool = True, verify_splits: list[str] | None = None):
        self.dir = Path(path)
        self.manifest = json.loads((self.dir / "MANIFEST.json").read_text())
        self.tokenizer = DoseTokenizer.load(self.dir / "tokenizer.json")
        if self.tokenizer.hash != self.manifest["tokenizer"]["hash"]:
            raise ValueError(f"{self.dir}: tokenizer hash mismatch")
        if verify:
            for s, info in self.manifest["splits"].items():
                if verify_splits is not None and s not in verify_splits:
                    continue
                got = sha256_file(self.dir / info["file"])
                if got != info["sha256"]:
                    raise ValueError(f"{self.dir}/{info['file']}: sha256 {got[:12]} != manifest {info['sha256'][:12]}")
        self.vocab_size = self.tokenizer.vocab_size
        self.bos_id = self.tokenizer.bos_id
        self.token_bytes = torch.tensor(self.tokenizer.token_byte_lengths(), dtype=torch.int64)
        self._splits: dict[str, np.memmap] = {}

    @property
    def hash(self) -> str:
        return f"{self.tokenizer.hash}:{self.manifest['splits']['train']['sha256'][:16]}"

    def has_split(self, name: str) -> bool:
        return name in self.manifest["splits"] and self.manifest["splits"][name]["n_tokens"] > 0

    def split(self, name: str) -> np.memmap:
        if name not in self._splits:
            self._splits[name] = np.memmap(self.dir / self.manifest["splits"][name]["file"], dtype=np.uint16, mode="r")
        return self._splits[name]

    def doc_bytes(self, split: str = "train") -> np.ndarray:
        return np.load(self.dir / f"{split}.docbytes.npy")

    def prefix(self, n_docs: int) -> tuple[int, int]:
        """(tokens, bytes) of the first n_docs training documents."""
        tokens = self.split("train")
        starts = np.flatnonzero(tokens == self.bos_id)
        end = starts[n_docs] if n_docs < len(starts) else len(tokens)
        return int(end), int(self.doc_bytes()[:n_docs].sum())

    def docs_for_bytes(self, n_bytes: int) -> int:
        """Smallest document prefix holding at least n_bytes of text."""
        cum = np.cumsum(self.doc_bytes())
        if n_bytes > cum[-1]:
            raise ValueError(f"{self.dir}: wanted {n_bytes:,} bytes, shard holds {int(cum[-1]):,}")
        return int(np.searchsorted(cum, n_bytes) + 1)
