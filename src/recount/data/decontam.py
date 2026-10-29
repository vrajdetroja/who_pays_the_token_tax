"""Drop training documents that share a 13-gram with FLORES (dev + devtest).

Text is normalized (NFKC, lowercase, every run of non-letter, non-number characters
becomes one space). Spaced scripts are compared in words; Han text, which has no spaces,
in characters, where 13 characters is roughly a 13-morpheme span.
"""

from __future__ import annotations

import unicodedata
from multiprocessing import get_context

import regex

_NON_WORD = regex.compile(r"[^\p{L}\p{M}\p{N}]+")
N = 13


def units(text: str, by_char: bool) -> list[str]:
    t = _NON_WORD.sub(" ", unicodedata.normalize("NFKC", text).lower()).strip()
    return [c for c in t if c != " "] if by_char else t.split()


def ngrams(text: str, by_char: bool, n: int = N) -> set[str]:
    u = units(text, by_char)
    j = "" if by_char else " "
    return {j.join(u[i : i + n]) for i in range(len(u) - n + 1)}


_REF: set[str] = set()
_BY_CHAR = False


def _init(ref: set[str], by_char: bool) -> None:
    global _REF, _BY_CHAR
    _REF, _BY_CHAR = ref, by_char


def _hit(text: str) -> bool:
    u = units(text, _BY_CHAR)
    j = "" if _BY_CHAR else " "
    return any(j.join(u[i : i + N]) in _REF for i in range(len(u) - N + 1))


def contaminated(texts: list[str], reference: list[str], by_char: bool, workers: int = 8) -> list[bool]:
    """For each text, whether it shares an n-gram with any reference sentence."""
    ref = set().union(*(ngrams(s, by_char) for s in reference))
    if not ref:
        return [False] * len(texts)
    with get_context("fork").Pool(workers, initializer=_init, initargs=(ref, by_char)) as pool:
        return pool.map(_hit, texts, chunksize=256)
