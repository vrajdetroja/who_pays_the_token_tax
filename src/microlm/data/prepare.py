"""Download, split, tokenize, and freeze a corpus.

    python -m microlm.data.prepare download   --source fineweb_edu
    python -m microlm.data.prepare tok-sample --source fineweb_edu --bytes 5e8
    python -m microlm.data.prepare tokenize   --source fineweb_edu --tokenizer tokenizers/bpe_8192.json
    python -m microlm.data.prepare verify     --data-dir data/fineweb_edu/bpe_8192

`tokenize` writes, under data/<source>/<tokenizer name>/:

    train.bin  val_iid.bin  val_long.bin  [val_ood.bin]   flat uint16 token streams
    tokenizer.json                                       copy of the tokenizer used
    MANIFEST.json                                        sha256, token/doc/byte counts

Each document is written as BOS followed by its tokens. Output directories are never
overwritten: a new tokenizer or split rule means a new directory with a new hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from ..tokenizer.bpe import Tokenizer, _encode_worker, _init_encode_worker
from ..utils import atomic_write_text, bounded_imap, git_sha, sha256_file
from . import sources as S

MANIFEST_VERSION = 1


class SplitWriter:
    """Appends uint16 documents to a .bin file, hashing as it goes."""

    def __init__(self, path: Path, bos_id: int):
        self.path = path
        self.f = open(path, "wb")
        self.sha = hashlib.sha256()
        self.bos = np.array([bos_id], dtype=np.uint16)
        self.n_tokens = self.n_docs = self.n_bytes = 0

    def add(self, ids: list[int], n_bytes: int) -> None:
        arr = np.concatenate([self.bos, np.asarray(ids, dtype=np.uint16)])
        buf = arr.tobytes()
        self.f.write(buf)
        self.sha.update(buf)
        self.n_tokens += len(arr)
        self.n_docs += 1
        self.n_bytes += n_bytes

    def close(self) -> dict:
        self.f.close()
        return {
            "file": self.path.name,
            "sha256": self.sha.hexdigest(),
            "n_tokens": self.n_tokens,
            "n_docs": self.n_docs,
            "n_bytes": self.n_bytes,
        }


def _encode_item(item: tuple[str, str]) -> tuple[str, list[int], int]:
    split, text = item
    return split, _encode_worker(text), len(text.encode("utf-8", "surrogatepass"))


def _encode_only(text: str) -> tuple[str, list[int], int]:
    return "val_ood", _encode_worker(text), len(text.encode("utf-8", "surrogatepass"))


def cmd_download(args) -> None:
    src = S.SOURCES[args.source]
    if args.max_files is not None:
        src = S.Source(**{**src.__dict__, "max_files": args.max_files})
    print(f"downloading {src.repo_id} -> {S.raw_dir(Path(args.data_root), src)}")
    S.download(src, Path(args.data_root))


def cmd_tok_sample(args) -> None:
    """The first N bytes of *train-split* documents, for training the tokenizer."""
    src = S.SOURCES[args.source]
    out = Path(args.data_root) / src.name / "tok_train_sample.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    target, seen, docs = int(args.bytes), 0, 0
    with open(out, "w") as f:
        for text in S.iter_documents(src, Path(args.data_root)):
            if S.assign_split(text) != "train":
                continue
            f.write(json.dumps({"text": text}) + "\n")
            seen += len(text.encode("utf-8", "surrogatepass"))
            docs += 1
            if seen >= target:
                break
    print(f"wrote {docs:,} docs / {seen / 1e6:.1f} MB to {out}")


def cmd_tokenize(args) -> None:
    src = S.SOURCES[args.source]
    data_root = Path(args.data_root)
    tok = Tokenizer.load(args.tokenizer)
    if tok.vocab_size > 65536:
        raise ValueError("uint16 storage needs vocab_size <= 65536")
    name = Path(args.tokenizer).stem
    out = data_root / src.name / name
    if out.exists():
        raise FileExistsError(f"{out} exists; frozen data is never overwritten. Delete it deliberately.")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copy(args.tokenizer, tmp / "tokenizer.json")

    limits = {
        "train_tokens": int(args.max_train_tokens) if args.max_train_tokens else None,
        "val_iid_bytes": int(args.val_iid_bytes),
        "val_long_docs": int(args.val_long_docs),
        "val_ood_bytes": int(args.val_ood_bytes),
    }
    writers = {s: SplitWriter(tmp / f"{s}.bin", tok.bos_id) for s in ("train", "val_iid", "val_long")}
    dropped = {"val_iid": 0, "val_long": 0, "train": 0}

    def full(split: str) -> bool:
        w = writers[split]
        if split == "train":
            return limits["train_tokens"] is not None and w.n_tokens >= limits["train_tokens"]
        if split == "val_iid":
            return w.n_bytes >= limits["val_iid_bytes"]
        return w.n_docs >= limits["val_long_docs"]

    def wanted():
        # Splitting is a cheap hash; encoding is not. A document whose split is already full
        # is dropped before it reaches a worker. This never changes the output: a split that
        # is full now is still full when the document's turn to be written comes.
        for text in S.iter_documents(src, data_root):
            split = S.assign_split(text)
            if full(split):
                dropped[split] += 1
            else:
                yield split, text

    t0 = time.time()
    workers = args.workers
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_encode_worker, initargs=(str(args.tokenizer),)) as pool:
        stream = bounded_imap(pool, _encode_item, wanted())
        for i, (split, ids, n_bytes) in enumerate(stream):
            if full(split):
                dropped[split] += 1  # never re-routed to another split
            else:
                writers[split].add(ids, n_bytes)
            if all(full(s) for s in writers):
                pool.terminate()
                break
            if i % 20000 == 0:
                rate = sum(w.n_bytes for w in writers.values()) / 1e6 / max(time.time() - t0, 1e-9)
                print(
                    f"  docs {i:>10,}  train {writers['train'].n_tokens / 1e6:8.1f}M tok  "
                    f"val_iid {writers['val_iid'].n_bytes / 1e6:6.1f} MB  "
                    f"val_long {writers['val_long'].n_docs:>4} docs  {rate:6.1f} MB/s",
                    flush=True,
                )
    splits = {s: w.close() for s, w in writers.items()}

    if args.ood_source:
        ood = S.SOURCES[args.ood_source]
        w = SplitWriter(tmp / "val_ood.bin", tok.bos_id)
        with ctx.Pool(workers, initializer=_init_encode_worker, initargs=(str(args.tokenizer),)) as pool:
            for _, ids, n_bytes in bounded_imap(pool, _encode_only, S.iter_documents(ood, data_root), chunksize=16):
                w.add(ids, n_bytes)
                if w.n_bytes >= limits["val_ood_bytes"]:
                    pool.terminate()
                    break
        splits["val_ood"] = w.close()
        splits["val_ood"]["source"] = ood.name

    for s, info in splits.items():
        print(f"  {s:>8}: {info['n_tokens']:>13,} tokens  {info['n_docs']:>9,} docs  "
              f"{info['n_bytes'] / max(info['n_tokens'], 1):.3f} bytes/token")
    if splits["val_iid"]["n_bytes"] < limits["val_iid_bytes"]:
        print(f"  WARNING: source exhausted before val_iid reached {limits['val_iid_bytes']:,} bytes")
    if splits["val_long"]["n_docs"] < limits["val_long_docs"]:
        print(f"  WARNING: only {splits['val_long']['n_docs']} val_long docs available")

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "source": src.__dict__ | {"local_files": [p.name for p in S.local_files(src, data_root)]},
        "tokenizer": {"name": name, "hash": tok.hash, "vocab_size": tok.vocab_size, "bos_id": tok.bos_id},
        "dtype": "uint16",
        "split_rule": {
            "hash": "blake2b-64(text) mod 1000",
            "val_band": S.VAL_BAND,
            "long_band": S.LONG_BAND,
            "long_doc_bytes": S.LONG_DOC_BYTES,
        },
        "limits": limits,
        "dropped_docs": dropped,
        "splits": splits,
        "git_sha": git_sha(),
        "seconds": round(time.time() - t0, 1),
    }
    atomic_write_text(tmp / "MANIFEST.json", json.dumps(manifest, indent=2))
    os.replace(tmp, out)
    print(f"froze {out}")


def verify_manifest(data_dir: str | Path, splits: list[str] | None = None) -> dict:
    """Recompute sha256 of every split (or the named ones) and the tokenizer hash."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "MANIFEST.json").read_text())
    tok = Tokenizer.load(data_dir / "tokenizer.json")
    if tok.hash != manifest["tokenizer"]["hash"]:
        raise ValueError(f"{data_dir}: tokenizer hash mismatch")
    for s, info in manifest["splits"].items():
        if splits is not None and s not in splits:
            continue
        got = sha256_file(data_dir / info["file"])
        if got != info["sha256"]:
            raise ValueError(f"{data_dir}/{info['file']}: sha256 {got[:12]} != manifest {info['sha256'][:12]}")
    return manifest


def cmd_verify(args) -> None:
    m = verify_manifest(args.data_dir)
    print(f"ok: {args.data_dir} ({', '.join(m['splits'])})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download")
    p.add_argument("--source", required=True, choices=sorted(S.SOURCES))
    p.add_argument("--max-files", type=int)
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("tok-sample")
    p.add_argument("--source", required=True, choices=sorted(S.SOURCES))
    p.add_argument("--bytes", type=float, default=5e8)
    p.set_defaults(fn=cmd_tok_sample)

    p = sub.add_parser("tokenize")
    p.add_argument("--source", required=True, choices=sorted(S.SOURCES))
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--max-train-tokens", type=float)
    p.add_argument("--val-iid-bytes", type=float, default=4e7, help="~10M tokens at 8k vocab")
    p.add_argument("--val-long-docs", type=float, default=500)
    p.add_argument("--ood-source", choices=sorted(S.SOURCES))
    p.add_argument("--val-ood-bytes", type=float, default=8e6)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_tokenize)

    p = sub.add_parser("verify")
    p.add_argument("--data-dir", required=True)
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
