"""End-to-end check on the CPU, with synthetic text: the whole pipeline, before any real training.

    python -m recount.smoke --root /tmp/recount-smoke

Synthetic text in two scripts goes through every stage the real data will: tokenizers and
dose search, frozen shards, the full main grid (two seeds, both model sizes, every branch,
the swap tokenizer) with a 2-layer width-64 model, evaluation under both context protocols
with every control, and the preregistered analysis. It checks the invariants the study
rests on, not the numbers, which mean nothing at this size.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np

from . import runs_table
from .analysis import analyze as A
from .analysis import decompose as D
from .study import Study, load_study
from .tok.build import build_shards, build_tokenizers
from .tok.shard import ShardData
from .train import plan
from .train.queue import run_queue
from .train.stream import DocStream

TAMIL_C = "கஙசஞடணதநபமயரலவழளறன"
TAMIL_V = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "்"]
CYRILLIC = "абвгдежзийклмнопрстуфхцчшщыьэюя"


def _lexicon(rng: random.Random, script: str, n: int) -> list[str]:
    words = set()
    while len(words) < n:
        if script == "Taml":
            w = "".join(rng.choice(TAMIL_C) + rng.choice(TAMIL_V) for _ in range(rng.randint(1, 4)))
        else:
            w = "".join(rng.choice(CYRILLIC) for _ in range(rng.randint(2, 9)))
        words.add(w)
    return sorted(words)


def _docs(rng: random.Random, lex: list[str], n_bytes: int, sentences: tuple[int, int] = (2, 30)) -> list[str]:
    weights = [1 / (i + 1) for i in range(len(lex))]  # Zipf
    out, seen = [], 0
    while seen < n_bytes:
        sents = []
        for _ in range(rng.randint(*sentences)):
            sents.append(" ".join(rng.choices(lex, weights, k=rng.randint(3, 14))).capitalize() + ".")
        doc = " ".join(sents)
        out.append(doc)
        seen += len(doc.encode())
    return out


def write_text(root: Path, lang: str, script: str, seed: int) -> None:
    rng = random.Random(seed)
    lex = _lexicon(rng, script, 1500)
    d = root / "data" / "text" / lang
    d.mkdir(parents=True, exist_ok=True)
    sizes = {"tok_sample": 400_000, "swap_sample": 400_000, "dose": 60_000, "heldout": 60_000, "content": 500_000}
    for name, n in sizes.items():
        with open(d / f"{name}.jsonl", "w") as f:
            for doc in _docs(rng, lex, n):
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
    with open(d / "flores.jsonl", "w") as f:
        for _ in range(20):
            k = rng.randint(3, 5)
            f.write(json.dumps({"text": "\n".join(_docs(rng, lex, 1, (1, 1))[0] for _ in range(k)),
                                "n_sentences": k}, ensure_ascii=False) + "\n")


def smoke_study() -> Study:
    s = load_study()
    s.languages = {"smk_Taml": {"name": "Smoke Tamil", "script": "Tamil"},
                   "smk_Cyrl": {"name": "Smoke Cyrillic", "script": "Cyrillic"}}
    s.tokenizer = s.tokenizer | {"vocab_size": 1024, "swap_language": "smk_Cyrl"}
    s.models = {"M": {"n_layer": 2, "d_model": 64, "n_head": 2, "english_equivalent_fraction": 1.0},
                "S": {"n_layer": 1, "d_model": 32, "n_head": 2, "english_equivalent_fraction": 0.5}}
    s.training = s.training | {"base_seq_len": 32, "windows_per_step": 16, "lr": 3e-3, "dtype": "fp32"}
    s.seeds = [0, 1]
    s.scout = s.scout | {"lr_language": "smk_Taml", "variance_languages": ["smk_Taml", "smk_Cyrl"],
                         "seeds": [0, 1], "swap_seeds": [0], "content_scale": 0.5}
    return s


def check(cond: bool, what: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        raise SystemExit(f"smoke failed: {what}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--keep", action="store_true", help="reuse an existing root instead of starting clean")
    args = ap.parse_args()
    root = Path(args.root)
    if root.exists() and not args.keep:
        shutil.rmtree(root)
    study = smoke_study()
    n_merges = study.tokenizer["vocab_size"] - 257

    print("== text, tokenizers, doses, shards")
    for i, (lang, script) in enumerate([("smk_Taml", "Taml"), ("smk_Cyrl", "Cyrl")]):
        if not (root / "data" / "text" / lang).exists():
            write_text(root, lang, script, seed=i)
        tok_dir = root / "data" / "tokenizers" / lang
        rep = build_tokenizers(lang, root / "data" / "text" / lang, tok_dir, study, n_merges)
        for lv, info in rep["levels"].items():
            check(abs(info["r_verify"] / float(lv) - 1) <= study.tokenizer["tolerance"],
                  f"{lang} r={lv}: verified r {info['r_verify']:.3f} (k = {info['active']})")
        build_shards(lang, root / "data" / "text" / lang, tok_dir, root / "data" / "shards", study)

    print("== common random numbers")
    shards = {lv: ShardData(plan.shard_dir(root, "smk_Taml", lv, "main")) for lv in study.levels}
    texts = {lv: [s.tokenizer.decode(DocStream(s.split("train"), s.bos_id, 32, seed=0).stream[:4000].tolist())]
             for lv, s in shards.items()}
    check(len({t[0][:300] for t in texts.values()}) == 1, "every level reads the same documents in the same order")
    a = shards[1.0].doc_bytes()
    check(all(np.array_equal(a, s.doc_bytes()) for s in shards.values()), "shards hold byte-identical content")

    print("== train and evaluate the main grid")
    jobs = plan.grid(study, root, "main")
    plan.dump(jobs, root / "queues" / "main.yaml")
    jobs = plan.load(root / "queues" / "main.yaml")
    n = run_queue(jobs, study, root, {"compile": False, "device": "cpu", "log_interval": 10_000,
                                      "micro_batch_size": 16})
    check(n == len(jobs), f"all {len(jobs)} jobs trained and evaluated")

    print("== analysis")
    runs = runs_table.load(root / "results" / "all_runs.csv")
    c = D.cells(runs)
    gap = (c["residual"] + c["phi_budget"] + c["phi_context"] - c["deployed"]).abs().max()
    check(gap < 1e-12, f"decomposition sums exactly (max gap {gap:.1e})")
    check(len(c) == 2 * 2 * 2 * 2, f"{len(c)} complete (lang, model, seed, level) units")
    res = A.run(runs, study)
    check("levels" in res["models"]["M"], "preregistered analysis runs end to end")
    nc2 = res["controls"].get("nc2_stride", {})
    check("max_abs_diff" in nc2, f"stride control evaluated (max |diff| {nc2.get('max_abs_diff', math.nan):.4f})")
    sh = D.control_effect(runs, "shuffle")
    check(len(sh) > 0 and sh["diff"].mean() > 0, f"context shuffle raises BPB (mean {sh['diff'].mean():+.4f})")
    dc = D.data_curve(runs)
    check(len(dc) == 8 and (dc["n_points"] == 4).all(), "data curve has 4 points per unit")

    print("== scout grid and its sealed report")
    scout = plan.grid(study, root, "scout")
    n = run_queue(scout, study, root, {"compile": False, "device": "cpu", "log_interval": 10_000,
                                       "micro_batch_size": 16}, runs_csv="results/scout_runs.csv")
    check(n == len(scout), f"all {len(scout)} scout jobs trained and evaluated")
    sruns = runs_table.load(root / "results" / "scout_runs.csv")
    res = A.run(sruns[~sruns["tokenizer"].str.contains(r"\+")], study)
    public, sealed = A.split_sealed(res)
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            keys.update(o)
            for v in o.values():
                walk(v)

    walk(public)
    check(not keys & A.SEALED_KEYS, "no level-contrast mean in the scout's public view")
    print("smoke passed")


if __name__ == "__main__":
    main()
