"""The token-id cache of prompt.Tokenizer: exact (same ids as tokenizing), bounded, thread-safe.

A small word-level tokenizer is built with the `tokenizers` library in a temporary directory,
so no checkpoint is needed. tests/integration/test_token_cache.py checks the real tokenizers
against the goldens.
"""

from __future__ import annotations

import json
import random
import threading

import pytest

from laya_apple.prompt import TokenCache, Tokenizer, prepare

WORDS = ["w%d" % i for i in range(200)] + "is it done which team how urgent level no yes".split()


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory):
    from tokenizers import Tokenizer as Backend
    from tokenizers import models, pre_tokenizers

    path = tmp_path_factory.mktemp("tokenizer")
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS])}
    backend = Backend(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.save(str(path / "tokenizer.json"))
    names = {"cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]", "mask_token": "[MASK]"}
    (path / "tokenizer_config.json").write_text(json.dumps(names))
    return path


class _Counting:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def encode(self, text, add_special_tokens):
        self.calls += 1
        return self.inner.encode(text, add_special_tokens=add_special_tokens)


def _text(rng, n):
    return " ".join(rng.choice(WORDS + ["unknown", "x"]) for _ in range(n))


def _uncached(tok_path):
    tok = Tokenizer(tok_path)
    tok.cache = None
    return tok


def test_cached_encode_returns_the_tokenizer_ids(tok_path):
    rng = random.Random(0)
    texts = [_text(rng, rng.randint(0, 60)) for _ in range(50)] + ["", "w1 [MASK] w2", "ünïcödé w3"]
    cached, plain = Tokenizer(tok_path), _uncached(tok_path)
    for _ in range(3):  # miss, then hits
        for t in texts:
            assert cached.encode(t) == plain.encode(t)


def test_a_hit_does_not_tokenize_again(tok_path):
    tok = Tokenizer(tok_path)
    tok.backend = counting = _Counting(tok.backend)
    first = tok.encode("is it done w1 w2")
    assert counting.calls == 1
    assert tok.encode("is it done w1 w2") == first
    assert counting.calls == 1
    tok.encode("is it done w1 w3")
    assert counting.calls == 2


def test_returned_ids_are_fresh_lists(tok_path):
    tok = Tokenizer(tok_path)
    ids = tok.encode("w1 w2 w3")
    ids.append(99)
    again = tok.encode("w1 w2 w3")
    assert again == ids[:-1]
    again.clear()
    assert tok.encode("w1 w2 w3") == ids[:-1]


def test_cache_is_bounded_by_entries_and_cost(monkeypatch):
    cache = TokenCache(entries=2, budget=1000)
    cache.put("a", [1])
    cache.put("b", [2])
    assert cache.get("a") == [1]  # "a" is now the most recent
    cache.put("c", [3])
    assert len(cache) == 2 and cache.get("b") is None and cache.get("a") == [1]
    cache = TokenCache(entries=100, budget=20)
    cache.put("x" * 8, [1, 2])  # cost 10
    cache.put("y" * 8, [1, 2])  # cost 10
    cache.put("z" * 8, [1, 2])  # evicts the oldest
    assert cache.cost == 20 and cache.get("x" * 8) is None and len(cache) == 2
    cache.put("big" * 10, [1])  # costs more than the whole budget: never kept
    assert cache.get("big" * 10) is None and cache.cost == 20
    cache.put("z" * 8, [7, 8])  # replacing an entry keeps the accounting right
    assert cache.cost == 20 and cache.get("z" * 8) == [7, 8]
    monkeypatch.setattr("laya_apple.prompt.TOKEN_CACHE_MAX_ENTRY", 5)
    cache = TokenCache()
    cache.put("long text", [1])
    assert len(cache) == 0
    disabled = TokenCache(entries=0)
    disabled.put("a", [1])
    assert disabled.get("a") is None


QUESTIONS = [
    {"type": "noul", "instructions": "is it done"},
    {"type": "choice", "instructions": "which team", "criteria": {"w1": "w2 w3", "w4": None}},
    {"type": "score", "instructions": "how urgent", "criteria": ["w5", "w6 w7", "w8"]},
]


def test_prepare_is_identical_with_and_without_the_cache(tok_path):
    rng = random.Random(1)
    cfg = {"max_len": 64, "head_max_len": 24}
    states = [_text(rng, 10), _text(rng, 200), {"k": _text(rng, 30)}, [_text(rng, 20) for _ in range(5)], ""]
    requests = []
    for state in states:  # the same context with different questions, and the reverse
        for i in range(len(QUESTIONS)):
            requests.append((state, {"q%d" % j: QUESTIONS[(i + j) % 3] for j in range(i + 1)}))
    cached, plain = Tokenizer(tok_path), _uncached(tok_path)
    for state, qs in requests + requests[::-1]:
        a, b = prepare(cached, cfg, state, qs), prepare(plain, cfg, state, qs)
        assert (a.items, a.truncated, a.question_ids) == (b.items, b.truncated, b.question_ids)
    assert len(cached.cache) > 0


def test_concurrent_encode_is_exact_under_eviction(tok_path):
    rng = random.Random(2)
    texts = [_text(rng, rng.randint(1, 40)) for _ in range(64)]
    expected = {t: _uncached(tok_path).encode(t) for t in texts}
    tok = Tokenizer(tok_path)
    tok.cache = TokenCache(entries=8)  # constant eviction
    errors = []

    def work(seed):
        r = random.Random(seed)
        for _ in range(500):
            t = r.choice(texts)
            if tok.encode(t) != expected[t]:
                errors.append(t)

    threads = [threading.Thread(target=work, args=(s,)) for s in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(tok.cache) <= 8
