"""XGLM's bits per FLORES-200 devtest sentence: the outcome of the XGLM check (PREREGISTRATION.md,
amendment 3; exploratory).

Each of the 1,012 sentences is scored alone after XGLM's `</s>`, under the model's own
tokenization: the sum over its tokens of -log p(token | `</s>`, the tokens before it). Batches
are right-padded, which leaves every real token's position and context as they are alone (XGLM
numbers positions 0, 1, ... whatever the mask says). The head runs only on the scored positions
and in chunks, because XGLM's 256,008-entry vocabulary makes a batch's full logits gigabytes.

The checkpoints are the Hugging Face ones at the revisions the amendment pins, downloaded once
(`make xglm-weights`, 4.6 GB) and loaded from disk. Both models share one tokenizer.

    python -I -m recount.eval.xglm --model-dir data/raw/xglm/xglm-564M --name 564M
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from microlm.utils import pick_device

from ..data import flores
from .canonical import append_csv

REVISIONS = {"564M": "f3059f01b98ccc877c673149e0178c0e957660f9", "1.7B": "d23a5e8e2164af31a84a26756b9b17f925143050"}
TRAIN_TOKENS = 500e9  # both model cards: "a balanced corpus ... totaling 500 billion sub-tokens"
LANGS = {  # ISO 639-1 -> (FLORES-200 code, share of training tokens after upsampling, name): the cards' table
    "en": ("eng_Latn", 0.3259, "English"), "ru": ("rus_Cyrl", 0.0602, "Russian"),
    "zh": ("zho_Hans", 0.0483, "Chinese"), "de": ("deu_Latn", 0.0363, "German"),
    "es": ("spa_Latn", 0.0353, "Spanish"), "fr": ("fra_Latn", 0.0313, "French"),
    "ja": ("jpn_Jpan", 0.0269, "Japanese"), "it": ("ita_Latn", 0.0171, "Italian"),
    "pt": ("por_Latn", 0.0297, "Portuguese"), "el": ("ell_Grek", 0.0233, "Greek"),
    "ko": ("kor_Hang", 0.0811, "Korean"), "fi": ("fin_Latn", 0.0681, "Finnish"),
    "id": ("ind_Latn", 0.0125, "Indonesian"), "tr": ("tur_Latn", 0.0101, "Turkish"),
    "ar": ("arb_Arab", 0.0099, "Arabic"), "vi": ("vie_Latn", 0.0091, "Vietnamese"),
    "th": ("tha_Thai", 0.0440, "Thai"), "bg": ("bul_Cyrl", 0.0393, "Bulgarian"),
    "ca": ("cat_Latn", 0.0287, "Catalan"), "hi": ("hin_Deva", 0.0140, "Hindi"),
    "et": ("est_Latn", 0.0133, "Estonian"), "bn": ("ben_Beng", 0.0066, "Bengali"),
    "ta": ("tam_Taml", 0.0060, "Tamil"), "ur": ("urd_Arab", 0.0055, "Urdu"),
    "sw": ("swh_Latn", 0.0037, "Swahili"), "te": ("tel_Telu", 0.0028, "Telugu"),
    "eu": ("eus_Latn", 0.0043, "Basque"), "my": ("mya_Mymr", 0.0030, "Burmese"),
    "ht": ("hat_Latn", 0.0035, "Haitian Creole"), "qu": ("quy_Latn", 0.0001, "Quechua"),
}


def load(model_dir: Path, device: str):
    from transformers import AutoTokenizer, XGLMForCausalLM  # the optional `xglm` extra

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = XGLMForCausalLM.from_pretrained(model_dir, dtype=torch.float32, local_files_only=True)
    return tok, model.to(device).eval()


def encode(tok, sentences: list[str]) -> list[list[int]]:
    """`</s>` then the sentence's tokens, as the tokenizer's own template builds them."""
    sep = tok.convert_tokens_to_ids("</s>")
    seqs = tok(sentences)["input_ids"]
    for s, ids in zip(sentences, seqs):
        if ids[0] != sep or sep in ids[1:] or len(ids) < 2:
            raise ValueError(f"unexpected encoding of {s[:40]!r}: {ids[:5]}")
    return seqs


@torch.inference_mode()
def score(model, seqs: list[list[int]], device, pad_id: int, max_tokens: int = 4096, head_rows: int = 512) -> np.ndarray:
    """Summed nats of every token after the first, per sequence. Longest batches first, so memory
    fails fast; a batch holds at most `max_tokens` positions, padding included."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    nats = np.zeros(len(seqs))
    i = 0
    while i < len(order):
        width = len(seqs[order[i]])
        j = i + max(1, max_tokens // width)
        rows = order[i:j]
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for r, k in enumerate(rows):
            ids[r, : len(seqs[k])], mask[r, : len(seqs[k])] = torch.tensor(seqs[k]), 1
        ids, mask = ids.to(device), mask.to(device)
        hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        scored = mask[:, 1:].bool()  # position t predicts token t + 1
        h, target = hidden[:, :-1][scored], ids[:, 1:][scored]
        owner = torch.arange(len(rows), device=device)[:, None].expand(-1, width - 1)[scored]
        nll = torch.empty(len(target), device=device)
        for a in range(0, len(target), head_rows):
            logits = model.lm_head(h[a : a + head_rows]).float()
            nll[a : a + head_rows] = torch.logsumexp(logits, -1) - logits.gather(1, target[a : a + head_rows, None])[:, 0]
        nats[rows] = torch.zeros(len(rows), device=device).index_add_(0, owner, nll).cpu().double().numpy()
        i = j
    return nats


def done_langs(path: Path, name: str) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["model", "lang"])
    counts = df[df["model"] == name].groupby("lang").size()
    return set(counts[counts == 1012].index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--name", required=True, choices=list(REVISIONS))
    ap.add_argument("--langs", nargs="*", default=list(LANGS))
    ap.add_argument("--flores", type=Path, default=flores.ROOT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-tokens", type=int, default=4096, help="positions per forward pass, padding included")
    ap.add_argument("--head-rows", type=int, default=512, help="positions per chunk of the 256,008-way head")
    ap.add_argument("--out", type=Path, default=Path("results/xglm/sentences.csv"))
    args = ap.parse_args()

    device = pick_device(args.device)
    tok, model = load(args.model_dir, device)
    skip = done_langs(args.out, args.name)  # a rerun resumes after the last finished language
    for iso in [l for l in args.langs if l not in skip]:
        t0, code = time.time(), LANGS[iso][0]
        sents = flores.load(code, "devtest", args.flores)
        seqs = encode(tok, sents)
        nats = score(model, seqs, device, tok.pad_token_id, args.max_tokens, args.head_rows)
        rows = [{"model": args.name, "lang": iso, "flores": code, "sentence": k, "n_bytes": len(s.encode("utf-8")),
                 "n_tokens": len(ids) - 1, "n_unk": ids[1:].count(tok.unk_token_id), "nats": round(float(n), 6)}
                for k, (s, ids, n) in enumerate(zip(sents, seqs, nats))]
        append_csv(args.out, rows)
        tokens = sum(r["n_tokens"] for r in rows)
        print(f"{args.name} {iso}: {nats.mean() / math.log(2):.1f} bits/sentence, {tokens} tokens, "
              f"{sum(r['n_unk'] for r in rows)} <unk>, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
