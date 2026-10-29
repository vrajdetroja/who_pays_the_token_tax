"""Sample FineWeb-2 (Penedo et al., 2025; ODC-By 1.0) documents by parquet row group.

Each parquet row group holds 1,000 documents, and row groups are clustered by Common
Crawl dump: in rus_Cyrl/train/000_00000.parquet, row group 0 is 2013-14 text and row
group 900 is 2022. The first N documents of a file are therefore a sample of one era of
the web. Instead, files are drawn at random (seeded) from the language's file list, row
groups at random within them, and row groups are read in parallel with HTTP range
requests until the byte target is met. Every document keeps its FineWeb-2 id, dump, and
URL, so the sample can be audited and reproduced.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

REPO = "HuggingFaceFW/fineweb-2"
COLUMNS = ["text", "id", "dump", "url"]


@dataclass(frozen=True)
class RowGroup:
    file: str
    index: int
    text_bytes: int  # uncompressed size of the text column (an estimate of UTF-8 bytes)


def list_files(lang: str, split: str) -> list[str]:
    from huggingface_hub import HfApi

    tree = HfApi().list_repo_tree(REPO, path_in_repo=f"data/{lang}/{split}", repo_type="dataset")
    return sorted(f.path for f in tree if f.path.endswith(".parquet"))


_local = threading.local()


def _open(path: str):
    """A per-thread, per-file open ParquetFile (opening reads the footer: seconds)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    cache = getattr(_local, "files", None)
    if cache is None:
        cache = _local.files = {}
        _local.fs = HfFileSystem()
    if path not in cache:
        f = _local.fs.open(f"datasets/{REPO}/{path}", "rb", block_size=1 << 20)
        cache[path] = pq.ParquetFile(f)
    return cache[path]


def footers(files: list[str], workers: int = 16) -> list[RowGroup]:
    def one(path: str) -> list[RowGroup]:
        md = _open(path).metadata
        out = []
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            text = next(rg.column(j) for j in range(rg.num_columns) if rg.column(j).path_in_schema == "text")
            out.append(RowGroup(path, i, text.total_uncompressed_size))
        return out

    with ThreadPoolExecutor(workers) as ex:
        return [rg for part in ex.map(one, files) for rg in part]


def _read(rg: RowGroup, retries: int = 5) -> list[dict]:
    for attempt in range(retries):
        try:
            tbl = _open(rg.file).read_row_group(rg.index, columns=COLUMNS)
            rows = tbl.to_pylist()
            for r in rows:
                r["file"], r["row_group"] = rg.file, rg.index
            return rows
        except Exception as e:  # network hiccups and 429s: back off and retry
            if attempt == retries - 1:
                raise
            print(f"  retry {rg.file}#{rg.index}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            _local.files = None
            time.sleep(2 ** attempt * 3)
    return []


def sample(lang: str, split: str, n_bytes: int, seed: int, max_files: int = 40, workers: int = 16,
           all_row_groups: bool = False) -> tuple[list[dict], dict]:
    """Documents from random row groups of random files until n_bytes of text.

    Row groups are drawn in a seeded random order and read in that order (results are
    re-ordered after the parallel reads), so the sample depends only on (lang, split, seed).
    """
    rng = random.Random(f"{lang}/{split}/{seed}")
    files = list_files(lang, split)
    chosen = sorted(rng.sample(files, min(max_files, len(files))))
    t0 = time.time()
    groups = footers(chosen, workers)
    available = sum(g.text_bytes for g in groups)
    rng.shuffle(groups)
    if not all_row_groups:
        need, take = int(n_bytes * 1.10), []  # 10% over the estimate, trimmed below
        for g in groups:
            take.append(g)
            need -= g.text_bytes
            if need <= 0:
                break
        groups = take
    print(f"  {lang}/{split}: {len(chosen)} of {len(files)} files, {len(groups)} row groups "
          f"({sum(g.text_bytes for g in groups) / 1e6:,.0f} MB est.) footers in {time.time() - t0:.0f}s", flush=True)
    results: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_read, g): i for i, g in enumerate(groups)}
        for k, fut in enumerate(as_completed(futs)):
            results[futs[fut]] = fut.result()
            if (k + 1) % 50 == 0:
                print(f"    {k + 1}/{len(groups)} row groups, {time.time() - t0:.0f}s", flush=True)
    docs = [d for i in range(len(groups)) for d in results[i]]
    meta = {"files_total": len(files), "files_chosen": chosen, "row_groups": [(g.file, g.index) for g in groups],
            "available_bytes_in_chosen_files": available,
            "available_bytes_estimate": int(available * len(files) / len(chosen)),
            "seconds": round(time.time() - t0, 1)}
    return docs, meta
