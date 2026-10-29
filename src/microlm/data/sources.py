"""Document sources and the split rule.

Every document is assigned to a split by a hash of its *content*, so the assignment does
not depend on file order, worker count, or which shards were downloaded, and exact
duplicates always land in the same split (no train/val leakage through duplicates).

    h = first 8 bytes of blake2b(text), as an integer, mod 1000
    h <  VAL_BAND                      -> val_iid pool
    h <  LONG_BAND and doc is long     -> val_long
    otherwise                          -> train

val_iid is the first `val_iid_bytes` of the val_iid pool, in source order. It is defined by
bytes, not tokens, so every tokenizer is evaluated on exactly the same text.
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

VAL_BAND = 10  # 1.0% of documents
LONG_BAND = 110  # 10% of long documents (outside the val band) are held out for val_long
LONG_DOC_BYTES = 24_000  # >= 4096 tokens (8x the 512 training context) for every vocab we train


def doc_hash(text: str) -> int:
    d = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(d, "big") % 1000


def assign_split(text: str) -> str:
    h = doc_hash(text)
    if h < VAL_BAND:
        return "val_iid"
    if h < LONG_BAND and len(text.encode("utf-8", "surrogatepass")) >= LONG_DOC_BYTES:
        return "val_long"
    return "train"


@dataclass(frozen=True)
class Source:
    name: str
    repo_id: str
    files: list[str] = field(default_factory=list)  # glob patterns within the repo
    text_field: str = "text"
    kind: str = "parquet"  # parquet | jsonl_gz
    max_files: int | None = None  # take only the first N matching files (sorted)


SOURCES: dict[str, Source] = {
    "fineweb_edu": Source(
        "fineweb_edu", "HuggingFaceFW/fineweb-edu", ["sample/10BT/*.parquet"], max_files=2
    ),
    "tinystories": Source("tinystories", "roneneldan/TinyStories", ["data/*.parquet"]),
    # Small local smoke-test corpus: the TinyStories validation shard only (~20 MB).
    "tinystories_mini": Source(
        "tinystories_mini", "roneneldan/TinyStories", ["data/validation-*.parquet"]
    ),
    # Out-of-distribution eval text: Python source code.
    "codeparrot_valid": Source(
        "codeparrot_valid",
        "codeparrot/codeparrot-clean-valid",
        ["*.json.gz"],
        text_field="content",
        kind="jsonl_gz",
    ),
}


def raw_dir(data_root: Path, source: Source) -> Path:
    return data_root / "raw" / source.name


def resolve_files(source: Source) -> list[str]:
    from huggingface_hub import HfApi

    all_files = HfApi().list_repo_files(source.repo_id, repo_type="dataset")
    files = sorted(f for f in all_files if any(fnmatch.fnmatch(f, p) for p in source.files))
    if source.max_files is not None:
        files = files[: source.max_files]
    if not files:
        raise FileNotFoundError(f"no files in {source.repo_id} match {source.files}")
    return files


def download(source: Source, data_root: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    out = raw_dir(data_root, source).resolve()
    paths = []
    for f in resolve_files(source):
        p = hf_hub_download(source.repo_id, f, repo_type="dataset", local_dir=out)
        paths.append(Path(p).resolve())
        print(f"  {f}  ({Path(p).stat().st_size / 1e9:.2f} GB)")
    (out / "FILES.json").write_text(json.dumps([str(p.relative_to(out)) for p in paths], indent=1))
    return paths


def local_files(source: Source, data_root: Path) -> list[Path]:
    out = raw_dir(data_root, source)
    listing = out / "FILES.json"
    if not listing.exists():
        raise FileNotFoundError(f"{out} has not been downloaded; run `prepare download` first")
    return [out / p for p in json.loads(listing.read_text())]


def iter_documents(source: Source, data_root: Path) -> Iterator[str]:
    """Yield document texts in a fixed order: sorted files, then row order within a file."""
    for path in local_files(source, data_root):
        if source.kind == "parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=4096, columns=[source.text_field]):
                for text in batch.column(0).to_pylist():
                    if text:
                        yield text
        elif source.kind == "jsonl_gz":
            with gzip.open(path, "rt") as f:
                for line in f:
                    text = json.loads(line)[source.text_field]
                    if text:
                        yield text
        else:
            raise ValueError(f"unknown source kind {source.kind!r}")
