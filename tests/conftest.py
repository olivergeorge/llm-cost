"""Shared pytest fixtures.

The plugin deliberately ships no bundled price data — users run
``llm cost refresh-prices`` once to download LiteLLM's catalog into a
user cache, and ``default_prices()`` reads from there. For tests we
plant a minimal synthetic cache covering the handful of models the
CLI/inline suites exercise, so they can assert on priced totals
without needing a live network fetch or a frozen upstream snapshot.

The fixture is autouse so that every test is isolated from whatever
real ``~/.config/llm-cost/prices.json`` exists on the developer's
machine — tests should never depend on (or pollute) that file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_cost import pricing
from llm_cost.pricing import default_prices


# $/1M figures kept here (readable) and divided once before writing the
# per-token cache. Tests assert totals against these numbers.
# Tuple shape: (input, output, audio_input, cache_read, reasoning). None
# slots mean "not set" (loader treats them as absent).
_TEST_PRICE_TABLE_PER_MILLION = {
    "claude-opus-4-6": (5.0, 25.0, None, None, None),
    "claude-sonnet-4-6": (3.0, 15.0, None, None, None),
    "claude-haiku-4-5": (0.8, 4.0, None, None, None),
    "gemini-3-flash-preview": (0.5, 3.0, None, None, None),
    # Gemini 2.5 flash mirrors LiteLLM today: audio $1/M vs text $0.30/M,
    # cache read $0.03/M, reasoning priced at the base output rate.
    "gemini-2.5-flash": (0.3, 2.5, 1.0, 0.03, 2.5),
    "gpt-4o": (2.5, 10.0, None, None, None),
}


def _price_spec(inp, outp, audio, cache, reasoning):
    spec = {
        "input_cost_per_token": inp / 1_000_000,
        "output_cost_per_token": outp / 1_000_000,
    }
    if audio is not None:
        spec["input_cost_per_audio_token"] = audio / 1_000_000
    if cache is not None:
        spec["cache_read_input_token_cost"] = cache / 1_000_000
    if reasoning is not None:
        spec["output_cost_per_reasoning_token"] = reasoning / 1_000_000
    return spec


@pytest.fixture(autouse=True)
def _test_price_cache(tmp_path: Path, monkeypatch):
    """Write a synthetic LiteLLM-shaped cache and point pricing at it."""
    cache_path = tmp_path / "llm-cost" / "prices.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                model: _price_spec(*rates)
                for model, rates in _TEST_PRICE_TABLE_PER_MILLION.items()
            }
        )
    )
    monkeypatch.setattr(pricing, "user_cache_path", lambda: cache_path)
    default_prices.cache_clear()
    yield cache_path
    default_prices.cache_clear()
