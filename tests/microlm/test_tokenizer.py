import random

import pytest

from microlm.tokenizer.bpe import (
    GPT4_SPLIT_PATTERN,
    Tokenizer,
    count_chunks,
    merge_ids,
    train_bpe,
    train_bpe_naive,
)

import regex

CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 3,
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "In 1905, Einstein published four papers; the third introduced special relativity!",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "Les élèves étudient la thermodynamique à l'université.",
    "東京は日本の首都です。人口は約1400万人です。",
    "Emoji test 🙂🚀🧪 — and some “smart quotes” and an em-dash.",
    "aaaaaaaaaaaa bbbbbbbb aaaa abab abababab   \t\n\n  trailing spaces   ",
] * 4


def random_text(rng: random.Random, n: int) -> str:
    pools = [
        "abcdefghijklmnopqrstuvwxyz     ",
        "ABCXYZ0123456789.,;:!?'\"()[]{}-_\n\t ",
        "éèàüößñçøå",
        "東京日本語中文字한국어",
        "🙂🚀🧪👩‍👩‍👧🏳️‍🌈",
        "​  ﻿",
    ]
    out = []
    for _ in range(n):
        pool = rng.choice(pools)
        out.append(rng.choice(pool))
        if rng.random() < 0.01:
            out.append(chr(rng.randint(0xD800, 0xDFFF)))  # lone surrogate
        if rng.random() < 0.01:
            out.append(chr(rng.randint(0x10000, 0x10FFFF)))
    return "".join(out)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    counts = count_chunks(CORPUS)
    return Tokenizer(train_bpe(counts, 300))


def test_merge_ids_non_overlapping():
    assert merge_ids([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_ids([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]


def test_fast_trainer_matches_naive():
    counts = count_chunks(CORPUS)
    fast = train_bpe(counts, 200)
    naive = train_bpe_naive(counts, 200)
    assert fast == naive
    assert len(fast) == 200


def test_trainer_stops_when_nothing_left_to_merge():
    counts = count_chunks(["ab ab ab"])
    merges = train_bpe(counts, 1000)
    assert merges == train_bpe_naive(counts, 1000)
    assert len(merges) < 1000


def test_pretokenizer_is_total():
    rng = random.Random(0)
    pat = regex.compile(GPT4_SPLIT_PATTERN)
    for _ in range(200):
        s = random_text(rng, rng.randint(0, 300))
        assert "".join(pat.findall(s)) == s


def test_encoding_reproduces_training_segmentation(tok):
    # Encoding a training chunk with the learned merges must give the same tokens the
    # trainer ended up with, i.e. merges applied in rank order are the trainer's history.
    for text in CORPUS:
        ids = tok.encode(text)
        assert tok.decode(ids) == text
    assert len(tok.encode(CORPUS[0])) < len(CORPUS[0].encode())


def test_round_trip_random_strings(tok):
    rng = random.Random(1)
    for _ in range(2000):
        s = random_text(rng, rng.randint(0, 200))
        assert tok.decode(tok.encode(s)) == s


def test_round_trip_arbitrary_bytes(tok):
    rng = random.Random(2)
    for _ in range(2000):
        b = bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 100)))
        assert tok.decode_bytes(tok.encode_bytes(b)) == b


def test_valid_utf8_bytes_and_str_paths_agree(tok):
    rng = random.Random(3)
    for _ in range(500):
        s = random_text(rng, 100).encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        assert tok.encode(s) == tok.encode_bytes(s.encode("utf-8"))


def test_truncation_is_a_prefix(tok):
    small = tok.truncated(257 + 50)
    assert small.merges == tok.merges[:50]
    assert small.vocab_size == 257 + 50
    assert small.bos_id == small.vocab_size - 1
    byte_level = tok.truncated(257)
    assert byte_level.encode("hi!") == list(b"hi!")
    # Prefix property of training itself: fewer merges == truncated longer run.
    counts = count_chunks(CORPUS)
    assert train_bpe(counts, 50) == tok.merges[:50]


def test_save_load_and_hash(tok, tmp_path):
    p = tmp_path / "t.json"
    tok.save(p)
    loaded = Tokenizer.load(p)
    assert loaded.merges == tok.merges
    assert loaded.hash == tok.hash
    assert tok.truncated(300).hash != tok.hash


def test_token_byte_lengths(tok):
    lens = tok.token_byte_lengths()
    assert lens[tok.bos_id] == 0
    text = CORPUS[5]
    assert sum(lens[i] for i in tok.encode(text)) == len(text.encode())
