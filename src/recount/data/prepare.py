"""Text for each study language: FineWeb-2 samples, decontaminated, split into the study's sets.

    python -I -m recount.data.prepare --lang eus_Latn rus_Cyrl arb_Arab cmn_Hani tam_Taml

Writes data/text/<lang>/:

    tok_sample.jsonl    200 MB from train, disjoint from everything below (tokenizer training)
    swap_sample.jsonl   another 200 MB, swap language only (negative control 1)
    content.jsonl       byte premium x 600 MB from train: the training content, canonical order
    dose.jsonl          2 MB x premium from test: what the dose search measures r on
    heldout.jsonl       2 MB x premium from test: the primary evaluation set
    flores.jsonl        FLORES-200 devtest, one record per article
    MANIFEST.json       sources, premium, availability, decontamination counts, sha256

Train documents are shuffled (seeded) before the split, so each set is a random draw from
the sampled row groups. Every set except flores is 13-gram decontaminated against FLORES
dev + devtest, and test documents identical to a sampled train document are dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from microlm.utils import atomic_write_text, git_sha, sha256_file

from ..study import Study, load_study
from . import decontam, fineweb2, flores, premiums


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8", "surrogatepass"))


def take(docs: list[dict], n_bytes: int, start: int) -> tuple[list[dict], int]:
    """Documents from `start` until at least n_bytes; returns them and the next index."""
    out, seen, i = [], 0, start
    while seen < n_bytes:
        if i >= len(docs):
            raise ValueError(f"ran out of documents: wanted {n_bytes:,} bytes, got {seen:,}")
        out.append(docs[i])
        seen += _nbytes(docs[i]["text"])
        i += 1
    return out, i


def write_jsonl(path: Path, docs: list[dict]) -> dict:
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({k: d[k] for k in ("text", "id", "dump", "url") if k in d} |
                               ({"n_sentences": d["n_sentences"]} if "n_sentences" in d else {}),
                               ensure_ascii=False) + "\n")
    return {"file": path.name, "docs": len(docs), "bytes": sum(_nbytes(d["text"]) for d in docs),
            "sha256": sha256_file(path)}


def prepare(lang: str, study: Study, root: Path, workers: int = 16) -> dict:
    out = root / "data" / "text" / lang
    if (out / "MANIFEST.json").exists():
        print(f"{lang}: {out} is prepared; delete it deliberately to redo")
        return json.loads((out / "MANIFEST.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    cfg, c = study.languages[lang], study.content
    t0 = time.time()
    bp = premiums.premium(lang)
    content_bytes = int(bp["premium"] * c["english_equivalent_bytes"])
    eval_bytes = int(bp["premium"] * c["heldout_english_equivalent_bytes"])
    tok_bytes = int(c["tokenizer_sample_bytes"])
    swap_bytes = tok_bytes if lang == study.tokenizer["swap_language"] else 0
    by_char = cfg["script"] == "Han"
    reference = flores.load(cfg["flores"], "dev") + flores.load(cfg["flores"], "devtest")

    # -- train split ------------------------------------------------------------------
    docs, meta = fineweb2.sample(lang, "train", int((content_bytes + tok_bytes + swap_bytes) * 1.03),
                                 c["sample_seed"], workers=workers)
    avail = meta["available_bytes_estimate"]
    if avail < c["availability_factor"] * content_bytes:
        raise ValueError(f"{lang}: ~{avail / 1e9:.1f} GB available < {c['availability_factor']}x the "
                         f"{content_bytes / 1e9:.2f} GB budget; use the fallback language")
    random.Random(f"{lang}/shuffle/{c['sample_seed']}").shuffle(docs)
    n_sampled = len(docs)
    hit = decontam.contaminated([d["text"] for d in docs], reference, by_char)
    dropped_train = sum(hit)
    docs = [d for d, h in zip(docs, hit) if not h]
    tok, i = take(docs, tok_bytes, 0)
    swap, i = take(docs, swap_bytes, i) if swap_bytes else ([], i)
    content, i = take(docs, content_bytes, i)
    train_hashes = {hashlib.blake2b(d["text"].encode(), digest_size=16).digest() for d in docs}

    # -- test split -------------------------------------------------------------------
    test, test_meta = fineweb2.sample(lang, "test", 0, c["sample_seed"], workers=workers, all_row_groups=True)
    random.Random(f"{lang}/test/{c['sample_seed']}").shuffle(test)
    hit = decontam.contaminated([d["text"] for d in test], reference, by_char)
    dup = [hashlib.blake2b(d["text"].encode(), digest_size=16).digest() in train_hashes for d in test]
    test = [d for d, h, u in zip(test, hit, dup) if not (h or u)]
    dose, j = take(test, eval_bytes, 0)
    heldout, j = take(test, eval_bytes, j)

    splits = {"tok_sample": tok, "content": content, "dose": dose, "heldout": heldout}
    if swap:
        splits["swap_sample"] = swap
    info = {name: write_jsonl(out / f"{name}.jsonl", d) for name, d in splits.items()}
    info["flores"] = write_jsonl(out / "flores.jsonl", flores.articles(cfg["flores"], "devtest"))
    manifest = {
        "lang": lang, "premium": bp, "content_bytes_target": content_bytes, "eval_bytes_target": eval_bytes,
        "splits": info,
        "decontamination": {"reference": f"FLORES-200 {cfg['flores']} dev+devtest", "n": decontam.N,
                            "unit": "character" if by_char else "word", "dropped_train": dropped_train,
                            "dropped_test": sum(hit), "dropped_test_duplicates": sum(dup)},
        "train_sample": meta | {"documents_sampled": n_sampled, "unused_documents": len(docs) - i},
        "test_sample": test_meta,
        "dumps_in_content": _dump_years(content),
        "git_sha": git_sha(), "seconds": round(time.time() - t0, 1),
    }
    atomic_write_text(out / "MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"{lang}: content {info['content']['bytes'] / 1e6:,.0f} MB ({info['content']['docs']:,} docs), "
          f"tok {info['tok_sample']['bytes'] / 1e6:,.0f} MB, heldout {info['heldout']['bytes'] / 1e6:.1f} MB, "
          f"decontam dropped {dropped_train} train / {sum(hit)} test docs, {time.time() - t0:.0f}s", flush=True)
    return manifest


def _dump_years(docs: list[dict]) -> dict:
    years: dict[str, int] = {}
    for d in docs:
        y = (d.get("dump") or "????????????")[8:12]
        years[y] = years.get(y, 0) + 1
    return dict(sorted(years.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", nargs="+", required=True)
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    study = load_study(args.study)
    for lang in args.lang:
        prepare(lang, study, Path(args.root), args.workers)


if __name__ == "__main__":
    main()
