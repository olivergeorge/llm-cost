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
_TEST_PRICE_TABLE_PER_MILLION = {
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (0.8, 4.0),
    "gemini-3-flash-preview": (0.5, 3.0),
    "gpt-4o": (2.5, 10.0),
}


@pytest.fixture(autouse=True)
def _test_price_cache(tmp_path: Path, monkeypatch):
    """Write a synthetic LiteLLM-shaped cache and point pricing at it."""
    cache_path = tmp_path / "llm-cost" / "prices.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                model: {
                    "input_cost_per_token": inp / 1_000_000,
                    "output_cost_per_token": outp / 1_000_000,
                }
                for model, (inp, outp) in _TEST_PRICE_TABLE_PER_MILLION.items()
            }
        )
    )
    monkeypatch.setattr(pricing, "user_cache_path", lambda: cache_path)
    default_prices.cache_clear()
    yield cache_path
    default_prices.cache_clear()
