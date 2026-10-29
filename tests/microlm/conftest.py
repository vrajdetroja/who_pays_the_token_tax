import json
import random

import numpy as np
import pytest

from microlm.data.prepare import SplitWriter
from microlm.tokenizer.bpe import Tokenizer, count_chunks, train_bpe

WORDS = "the a cat dog sat ran on under big small red blue house tree and then quickly slowly".split()


def _doc(rng: random.Random, n_sent: int) -> str:
    out = []
    for _ in range(n_sent):
        out.append(" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 9))).capitalize() + ".")
    return " ".join(out)


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    """A frozen data dir in the real on-disk format: tokenizer, .bin splits, MANIFEST.json."""
    d = tmp_path_factory.mktemp("data") / "synth"
    d.mkdir()
    rng = random.Random(0)
    docs = {"train": [_doc(rng, 8) for _ in range(1500)],
            "val_iid": [_doc(rng, 8) for _ in range(60)],
            "val_long": [_doc(rng, 200) for _ in range(4)]}
    tok = Tokenizer(train_bpe(count_chunks(docs["train"][:200]), 300 - 257))
    tok.save(d / "tokenizer.json")
    splits = {}
    for name, texts in docs.items():
        w = SplitWriter(d / f"{name}.bin", tok.bos_id)
        for t in texts:
            w.add(tok.encode(t), len(t.encode()))
        splits[name] = w.close()
    manifest = {
        "tokenizer": {"name": "synth", "hash": tok.hash, "vocab_size": tok.vocab_size, "bos_id": tok.bos_id},
        "splits": splits,
    }
    (d / "MANIFEST.json").write_text(json.dumps(manifest))
    return d


@pytest.fixture(autouse=True)
def _project1_cwd(monkeypatch):
    """The vendored tests load project 1's configs by relative path (configs/baseline.yaml).
    Copies of the three they use live in fixtures/; run each test from there."""
    from pathlib import Path

    monkeypatch.chdir(Path(__file__).parent / "fixtures")
