"""The token-id cache on the real checkpoint tokenizers: every golden case, prepared cold and
then again from a warm cache in both orders, gives upstream's exact prompt items."""

from __future__ import annotations

import pytest

from laya_apple.prompt import Tokenizer, prepare

pytestmark = pytest.mark.integration


def test_cached_prepare_reproduces_the_golden_items(checkpoint, config, golden):
    cached = Tokenizer(checkpoint / "tokenizer")
    plain = Tokenizer(checkpoint / "tokenizer")
    plain.cache = None
    cases = golden["cases"]
    for case in cases + cases[::-1] + cases:
        items = prepare(cached, config, case["state"], case["questions"]).items
        assert items == case["items"], case["name"]
        assert items == prepare(plain, config, case["state"], case["questions"]).items, case["name"]
    assert len(cached.cache) > 0
