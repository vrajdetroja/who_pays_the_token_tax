"""Tokenizers, fertility doses, and shards for one language (all CPU).

    python -m recount.tok.build --lang rus_Cyrl

Reads data/text/<lang>/ (written by recount.data.prepare):

    tok_sample.jsonl    tokenizer training text, disjoint from the content
    swap_sample.jsonl   a second disjoint sample (negative control 1 only)
    dose.jsonl          held-out text the dose search measures r on
    heldout.jsonl       held-out evaluation text: r is verified here, never searched
    content.jsonl       training content, canonical order
    flores.jsonl        FLORES+ devtest articles (optional)

and writes data/tokenizers/<lang>/{full,r1.5,r2.25,swap_r1.5}.json, data/shards/<lang>/<level>/,
and data/tokenizers/<lang>/dose_report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from microlm.utils import canonical_json

from ..study import Study, load_study
from . import shard
from .dose import DoseTokenizer, search_active, tokens_per_byte, train_full


def read_texts(path: Path, max_bytes: int | None = None) -> list[str]:
    out, seen = [], 0
    for text in shard.iter_jsonl(path):
        out.append(text)
        seen += len(text.encode("utf-8", "surrogatepass"))
        if max_bytes is not None and seen >= max_bytes:
            break
    return out


def build_tokenizers(lang: str, text_dir: Path, tok_dir: Path, study: Study, n_merges: int) -> dict:
    """Full tokenizer, dose levels by binary search on dose.jsonl, verification on
    heldout.jsonl, and the swap tokenizer for negative control 1."""
    tok_dir.mkdir(parents=True, exist_ok=True)
    tcfg = study.tokenizer
    dose_texts = read_texts(text_dir / "dose.jsonl")
    verify_texts = read_texts(text_dir / "heldout.jsonl")
    report: dict = {"lang": lang, "n_merges": n_merges, "levels": {}}

    def train(name: str) -> DoseTokenizer:
        path = tok_dir / f"{name}.json"
        if path.exists():
            return DoseTokenizer.load(path)
        t0 = time.time()
        sample = text_dir / f"{'tok' if name == 'full' else 'swap'}_sample.jsonl"
        tok = train_full(shard.iter_jsonl(sample), n_merges=n_merges)
        tok.meta |= {"lang": lang, "trained_on": sample.name, "train_seconds": round(time.time() - t0, 1)}
        tok.save(path)
        print(f"  {lang} {name}: {n_merges} merges in {time.time() - t0:.0f}s")
        return tok

    full = train("full")
    base_dose, base_verify = tokens_per_byte(full, dose_texts), tokens_per_byte(full, verify_texts)
    for level in tcfg["levels"]:
        k, r = search_active(full, dose_texts, level) if level > 1 else (full.active, 1.0)
        r_verify = tokens_per_byte(full.with_active(k), verify_texts) / base_verify
        if abs(r_verify / level - 1) > tcfg["tolerance"]:
            raise ValueError(f"{lang} r={level}: verified r = {r_verify:.4f} is outside +/-{tcfg['tolerance']:.0%}")
        tok = full.with_active(k, level=level, r_search=r, r_measured=r_verify, tokenizer="main")
        tok.save(tok_dir / f"r{level:g}.json")
        report["levels"][f"{level:g}"] = {"active": k, "r_search": r, "r_verify": r_verify,
                                          "bytes_per_token": 1 / (base_verify * r_verify)}
    if lang == tcfg["swap_language"]:
        level = tcfg["swap_level"]
        swap_full = train("swap_full")
        sb = tokens_per_byte(swap_full, dose_texts)
        # Same r means the same tokens per byte as the *main* tokenizer's level: target the
        # swap tokenizer's own ratio that lands on main's tokens/byte at that level.
        target = level * base_dose / sb
        k, r = search_active(swap_full, dose_texts, target)
        r_main = tokens_per_byte(swap_full.with_active(k), verify_texts) / base_verify
        if abs(r_main / level - 1) > tcfg["swap_tolerance"]:
            raise ValueError(f"{lang} swap: r = {r_main:.4f} is outside +/-{tcfg['swap_tolerance']:.0%} of {level}")
        tok = swap_full.with_active(k, level=level, r_search=r * sb / base_dose, r_measured=r_main, tokenizer="swap")
        tok.save(tok_dir / f"swap_r{level:g}.json")
        report["swap"] = {"active": k, "r_verify": r_main} | shared_tokens(
            full, report["levels"][f"{level:g}"]["active"], swap_full, k)
    (tok_dir / "dose_report.json").write_text(json.dumps(report, indent=2))
    return report


def shared_tokens(a: DoseTokenizer, ka: int, b: DoseTokenizer, kb: int) -> dict:
    """How many of the merged tokens two dose tokenizers can produce are the same byte
    strings. Merges are compared by the bytes they create: ids mean different things in
    tokenizers trained separately."""
    ta = {a.vocab_bytes[256 + i] for i in range(ka)}
    tb = {b.vocab_bytes[256 + i] for i in range(kb)}
    return {"shared_tokens": len(ta & tb), "shared_fraction": len(ta & tb) / max(len(tb), 1)}


def build_shards(lang: str, text_dir: Path, tok_dir: Path, shard_root: Path, study: Study) -> None:
    splits = {"train": text_dir / "content.jsonl", "heldout": text_dir / "heldout.jsonl"}
    if (text_dir / "flores.jsonl").exists():
        splits["flores"] = text_dir / "flores.jsonl"
    names = [f"r{lv:g}" for lv in study.tokenizer["levels"]]
    if lang == study.tokenizer["swap_language"]:
        names.append(f"swap_r{study.tokenizer['swap_level']:g}")
    for name in names:
        out = shard_root / lang / name
        if out.exists():
            print(f"  {out} exists, skipping")
            continue
        tok = DoseTokenizer.load(tok_dir / f"{name}.json")
        shard.build(out, tok, splits, extra={"lang": lang, "text_dir": str(text_dir)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", nargs="+", required=True)
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--skip-shards", action="store_true")
    args = ap.parse_args()
    study = load_study(args.study)
    root = Path(args.root)
    n_merges = study.tokenizer["vocab_size"] - 257
    for lang in args.lang:
        text_dir = root / "data" / "text" / lang
        tok_dir = root / "data" / "tokenizers" / lang
        report = build_tokenizers(lang, text_dir, tok_dir, study, n_merges)
        print(canonical_json(report))
        if not args.skip_shards:
            build_shards(lang, text_dir, tok_dir, root / "data" / "shards", study)


if __name__ == "__main__":
    main()
