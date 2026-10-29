import random

import pytest
import regex

from recount.tok.dose import MARK_AWARE_PATTERN, DoseTokenizer, search_active, tokens_per_byte, train_full

WORDS = {
    "tam": "தமிழ் மொழி ஒரு திராவிட மொழியாகும் இந்தியா இலங்கை சிங்கப்பூர் மக்கள் பேசுகின்றனர் அரசு கல்வி".split(),
    "cmn": "中文 分词 测试 我们 今天 去 北京 学习 语言 模型 数据 的 是 在 有 和 人 这 中国".split(),
    "rus": "привет мир это тест язык модель данные город страна время человек работа".split(),
    "arb": "اللغة العربية الفصحى هي لغة رسمية في كثير من الدول العربية والعالم".split(),
    "eus": "euskara hizkuntza da Euskal Herrian hitz egiten dena etxean eskolan lanean".split(),
}


def _corpus(n_docs: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        lang = rng.choice(list(WORDS))
        sep = "" if lang == "cmn" and rng.random() < 0.5 else " "
        sents = [sep.join(rng.choice(WORDS[lang]) for _ in range(rng.randint(4, 12))) for _ in range(rng.randint(1, 4))]
        docs.append(("。" if lang == "cmn" else ". ").join(sents) + rng.choice(["", "\n", " 2024", "!"]))
    return docs


@pytest.fixture(scope="module")
def full():
    return train_full(_corpus(3000, 0), n_merges=400)


def test_mark_aware_pattern_keeps_tamil_words_whole():
    assert regex.findall(MARK_AWARE_PATTERN, "தமிழ் மொழி") == ["தமிழ்", " மொழி"]


def test_id_space_is_fixed_across_levels(full):
    for k in (0, 50, 400):
        t = full.with_active(k)
        assert t.vocab_size == 257 + 400 and t.bos_id == 256 + 400
        assert t.unreachable_ids() == list(range(256 + k, 656))


def test_fast_encoder_matches_the_reference_encoder(full):
    texts = _corpus(300, 1) + ["mixed 中文 and தமிழ் text, 123456!", "  leading spaces\n\nnew lines\t tabs", "emoji 🙂 ok"]
    for k in (0, 7, 120, 400):
        t = full.with_active(k)
        for x in texts:
            assert t.encode_fast(x) == t.encode(x), (k, x)


def test_round_trip_and_unreachable_ids_never_appear(full):
    t = full.with_active(120)
    for x in _corpus(100, 2):
        ids = t.encode_fast(x)
        assert t.decode(ids) == x
        assert max(ids) < 256 + 120


def test_dose_search_hits_its_target_and_fertility_is_monotone(full):
    texts = _corpus(400, 3)
    rs = [tokens_per_byte(full.with_active(k), texts) for k in (0, 50, 100, 200, 400)]
    assert rs == sorted(rs, reverse=True)
    k, r = search_active(full, texts, 1.5)
    assert abs(r - 1.5) < 0.05 and 0 < k < 400


def test_save_load_round_trip(full, tmp_path):
    t = full.with_active(77, level=1.5)
    t.save(tmp_path / "t.json")
    u = DoseTokenizer.load(tmp_path / "t.json")
    assert u.hash == t.hash and u.active == 77 and u.meta["level"] == 1.5
    assert u.hash != full.hash
