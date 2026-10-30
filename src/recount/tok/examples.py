"""What the dose levels do to real text, for the appendix.

    python -I -m recount.tok.examples --out results/appendix/segmentation.md

One FLORES-200 devtest sentence, the same sentence in every language (FLORES is parallel), is
cut by each level's tokenizer. Tokens are separated by a middle dot, a leading space shows as
▁, and a token that is not whole UTF-8 characters (a byte fragment of a character) shows as
its bytes in hex inside ⟨⟩. A second table measures each level's dose on the whole FLORES-200
devtest beside the held-out r the levels were built to (dose_report.json), so the dose is
checked on text the tokenizers never saw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..analysis.appendix import LANG_NAMES
from ..data import flores
from ..study import load_study
from .dose import DoseTokenizer

SENTENCE = 29  # "The find also grants insight into the evolution of feathers in birds."
FLORES_LICENSE = "FLORES-200 (NLLB Team, 2024), CC-BY-SA 4.0"


def show(tok: DoseTokenizer, ids: list[int]) -> str:
    parts = []
    for i in ids:
        b = tok.vocab_bytes[i]
        try:
            parts.append(b.decode("utf-8").replace(" ", "▁"))
        except UnicodeDecodeError:
            parts.append("⟨" + " ".join(f"{x:02x}" for x in b) + "⟩")
    return "·".join(parts)


def tokens_per_byte(tok: DoseTokenizer, texts: list[str]) -> float:
    return sum(map(len, tok.encode_batch_fast(texts))) / sum(len(t.encode("utf-8")) for t in texts)


def build(study, root: Path = Path("data/tokenizers"), sentence: int = SENTENCE) -> tuple[str, list[dict]]:
    english = flores.load("eng_Latn")[sentence]
    sections, rows = [], []
    for lang, info in study.languages.items():
        text = flores.load(info["flores"])[sentence]
        devtest = flores.load(info["flores"])
        report = json.loads((root / lang / "dose_report.json").read_text())
        lines = []
        base = None
        for level in study.levels:
            tok = DoseTokenizer.load(root / lang / f"r{level:g}.json")
            ids = tok.encode_fast(text)
            assert b"".join(tok.vocab_bytes[i] for i in ids) == text.encode("utf-8"), "segmentation lost bytes"
            tpb = tokens_per_byte(tok, devtest)
            base = tpb if level == 1.0 else base
            rows.append({"lang": lang, "name": LANG_NAMES[lang], "level": level, "active_merges": tok.active,
                         "r_heldout": report["levels"][f"{level:g}"]["r_verify"], "r_flores": tpb / base,
                         "sentence_tokens": len(ids), "sentence_bytes": len(text.encode("utf-8"))})
            lines.append(f"r = {level:<4g} {len(ids):>3} tokens  {show(tok, ids)}")
        sections.append(f"\n### {LANG_NAMES[lang]} ({lang}; {len(text.encode('utf-8'))} bytes)\n\n```text\n"
                        + "\n".join(lines) + "\n```\n")
    head = (f"# Segmentation at each dose level\n\nFLORES-200 devtest sentence {sentence} ({FLORES_LICENSE}); "
            f"in English: \"{english}\"\n\nTokens are separated by ·, a leading space is ▁, and a token that "
            "splits a character shows its bytes in hex inside ⟨⟩. Each level keeps the first k merges of the "
            "language's 16,384-entry tokenizer (k in the table below).\n")
    return head + "".join(sections), rows


def dose_table_md(rows: list[dict]) -> str:
    lines = ["\n## The dose on FLORES-200 devtest\n",
             "Tokens per byte on all 1,012 devtest sentences, relative to the language's full tokenizer, beside "
             "the held-out FineWeb-2 r each level was built to (±3%). FLORES is out of domain for every "
             "tokenizer (news, travel and children's-encyclopedia sentences), so it shows how far the dose carries "
             "beyond the text it was built on.\n",
             "| language | level | merges kept | r, held-out | r, FLORES devtest |", "|---|---|---|---|---|"]
    lines += [f"| {r['name']} | {r['level']:g} | {r['active_merges']:,} | {r['r_heldout']:.3f} | {r['r_flores']:.3f} |"
              for r in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--sentence", type=int, default=SENTENCE)
    ap.add_argument("--out", default="results/appendix/segmentation.md")
    args = ap.parse_args()
    text, rows = build(load_study(args.study), sentence=args.sentence)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + dose_table_md(rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
