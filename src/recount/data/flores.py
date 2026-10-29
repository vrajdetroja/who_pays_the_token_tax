"""FLORES-200 (NLLB Team, 2024; CC-BY-SA 4.0): 2,009 English Wikinews / Wikijunior /
Wikivoyage sentences (997 dev + 1,012 devtest), professionally translated.

Used three ways: the secondary evaluation set (devtest, scored article by article in bits
per sentence), the decontamination reference (dev + devtest), and Table 1's token premiums.
Sentences are grouped into articles by the metadata's source URL; consecutive sentences
of one article stay together, so a model scores each sentence with its article's context.

The maintained successor, FLORES+ (openlanguagedata/flores_plus), is gated on Hugging Face
and names Mandarin cmn_Hans instead of zho_Hans; `load` takes either release's codes.
"""

from __future__ import annotations

import csv
from itertools import groupby
from pathlib import Path

ROOT = Path("data/raw/flores200/extracted/flores200_dataset")
PLUS_TO_200 = {"cmn_Hans": "zho_Hans", "cmn_Hant": "zho_Hant"}


def load(lang: str, split: str = "devtest", root: Path = ROOT) -> list[str]:
    code = PLUS_TO_200.get(lang, lang)
    path = root / split / f"{code}.{split}"
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    expected = {"dev": 997, "devtest": 1012}[split]
    if len(lines) != expected:
        raise ValueError(f"{path}: {len(lines)} sentences, expected {expected}")
    return lines


def urls(split: str = "devtest", root: Path = ROOT) -> list[str]:
    with open(root / f"metadata_{split}.tsv", newline="", encoding="utf-8") as f:
        return [row["URL"] for row in csv.DictReader(f, delimiter="\t")]


def articles(lang: str, split: str = "devtest", root: Path = ROOT, sep: str = "\n") -> list[dict]:
    """Consecutive sentences sharing a source URL, joined by `sep`."""
    sents, srcs = load(lang, split, root), urls(split, root)
    out, i = [], 0
    for url, group in groupby(srcs):
        n = len(list(group))
        out.append({"text": sep.join(sents[i : i + n]), "n_sentences": n, "url": url})
        i += n
    return out


def premium(lang: str, reference: str = "eng_Latn", root: Path = ROOT) -> float:
    """Bytes of devtest in `lang` over bytes in `reference`: a direct byte premium."""
    b = lambda code: sum(len(s.encode("utf-8")) for s in load(code, "devtest", root))  # noqa: E731
    return b(lang) / b(reference)
