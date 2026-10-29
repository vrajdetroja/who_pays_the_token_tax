"""Table 1, the stakes: what current tokenizers charge each language (CPU only).

    python -m recount.table1 --out results/table1

For each language and tokenizer, on the 1,012 parallel sentences of FLORES-200 devtest:

    premium        tokens(language) / tokens(English): the cost multiplier, whatever the price
    window_en      English-equivalent content of a 128k-token window = 128,000 / premium
    dose_r         (study languages, once their tokenizers exist) where the production
                   tokenizer sits on the study's dose axis: its tokens per byte over the
                   language's own full 16,384-entry tokenizer's

Sentences are encoded one at a time without special tokens. Tokenizers that cannot be
loaded (gated repositories without an accepted license) are reported and skipped.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from .data import flores

LANGS = {
    "eng_Latn": "English", "eus_Latn": "Basque", "rus_Cyrl": "Russian", "arb_Arab": "Arabic (MSA)",
    "zho_Hans": "Chinese (Simplified)", "tam_Taml": "Tamil", "ben_Beng": "Bengali", "hin_Deva": "Hindi",
    "amh_Ethi": "Amharic", "tha_Thai": "Thai", "mya_Mymr": "Burmese", "yor_Latn": "Yoruba",
}
STUDY = {"eus_Latn": "eus_Latn", "rus_Cyrl": "rus_Cyrl", "arb_Arab": "arb_Arab", "zho_Hans": "cmn_Hani",
         "tam_Taml": "tam_Taml"}
TOKENIZERS = {
    "o200k_base (GPT-4o/4.1/5)": ("tiktoken", "o200k_base"),
    "Llama 3": ("hf", "meta-llama/Llama-3.2-1B"),
    "Gemma 3": ("hf", "google/gemma-3-1b-pt"),
    "Qwen3": ("hf", "Qwen/Qwen3-8B"),
    "Mistral Nemo (Tekken)": ("hf", "mistralai/Mistral-Nemo-Base-2407"),
    "DeepSeek-V3": ("hf", "deepseek-ai/DeepSeek-V3"),
    "BLOOM (2022)": ("hf", "bigscience/bloom"),
    "XGLM (2021)": ("hf", "facebook/xglm-564M"),
}


def load_tokenizer(kind: str, name: str):
    """A function text -> number of tokens."""
    if kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(name)
        return lambda s: len(enc.encode(s, disallowed_special=()))
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(name)
    return lambda s: len(tok.encode(s, add_special_tokens=False).ids)


def count(n_tokens, sentences: list[str]) -> int:
    return sum(n_tokens(s) for s in sentences)


def build(langs: dict, tokenizers: dict, root: Path) -> tuple[pd.DataFrame, list[str]]:
    sents = {code: flores.load(code, "devtest") for code in langs}
    nbytes = {code: sum(len(s.encode()) for s in ss) for code, ss in sents.items()}
    rows, skipped = [], []
    ours = {}
    for code, study_code in STUDY.items():
        p = root / "data" / "tokenizers" / study_code / "full.json"
        if p.exists() and code in langs:
            from .tok.dose import DoseTokenizer

            t = DoseTokenizer.load(p)
            ours[code] = sum(len(x) for x in t.encode_batch_fast(sents[code])) / nbytes[code]
    for label, (kind, name) in tokenizers.items():
        try:
            n_tokens = load_tokenizer(kind, name)
        except Exception as e:  # gated or unavailable
            skipped.append(f"{label} ({name}): {type(e).__name__}: {str(e).splitlines()[0][:100]}")
            continue
        eng = count(n_tokens, sents["eng_Latn"])
        for code, ss in sents.items():
            n = count(n_tokens, ss)
            premium = n / eng
            row = {"lang": code, "language": langs[code], "tokenizer": label, "source": name, "tokens": n,
                   "bytes": nbytes[code], "premium": premium, "window_en_k": 128 / premium,
                   "tokens_per_byte": n / nbytes[code]}
            if code in ours:
                row["dose_r"] = (n / nbytes[code]) / ours[code]
            rows.append(row)
        print(f"  {label}: done", flush=True)
    return pd.DataFrame(rows), skipped


def to_markdown(df: pd.DataFrame, skipped: list[str], when: str) -> str:
    wide = df.pivot_table(index=["lang", "language"], columns="tokenizer", values="premium", sort=False)
    wide["mean"] = wide.mean(axis=1)
    wide = wide.sort_values("mean")
    lines = [f"# Table 1: token premium vs. English on FLORES-200 devtest ({when})", "",
             "Premium = tokens for the same 1,012 sentences / tokens in English; it is also the cost "
             "multiplier at any per-token price. A 128k-token window holds 128k / premium tokens' worth "
             "of English-equivalent content.", ""]
    cols = list(wide.columns)
    lines.append("| Language | " + " | ".join(cols) + " | 128k window holds |")
    lines.append("|---" * (len(cols) + 2) + "|")
    for (code, name), r in wide.iterrows():
        cells = " | ".join(f"{r[c]:.2f}" for c in cols)
        lines.append(f"| {name} | {cells} | {128 / r['mean']:.0f}k |")
    if "dose_r" in df:
        d = df.dropna(subset=["dose_r"]).pivot_table(index="language", columns="tokenizer", values="dose_r")
        lines += ["", "Position on the study's dose axis (tokens per byte relative to the language's own "
                  "16,384-entry monolingual tokenizer; the study's levels are 1.0, 1.5, 2.25):", "",
                  "| Language | " + " | ".join(d.columns) + " |", "|---" * (len(d.columns) + 1) + "|"]
        lines += [f"| {lang} | " + " | ".join(f"{v:.2f}" for v in r) + " |" for lang, r in d.iterrows()]
    if skipped:
        lines += ["", "Not included (could not be loaded):", ""] + [f"- {s}" for s in skipped]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/table1")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    when = date.today().isoformat()
    df, skipped = build(LANGS, TOKENIZERS, Path(args.root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.assign(date=when).to_csv(out.with_suffix(".csv"), index=False)
    md = to_markdown(df, skipped, when)
    out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
