from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_cost import pricing
from llm_cost.pricing import (
    Price,
    TokenBreakdown,
    _canonical,
    default_prices,
    load_prices,
    refresh_prices,
    resolve,
)


# Per-million USD is how providers publish rates — convert once and keep
# tests readable at the call site.
def _m(input_per_m: float, output_per_m: float) -> Price:
    return Price(input_per_m / 1_000_000, output_per_m / 1_000_000)


def _rich(input_per_m, output_per_m, audio_per_m=None, cache_per_m=None, reasoning_per_m=None):
    """Build a Price with optional audio/cache/reasoning rates set."""
    def per_token(v):
        return None if v is None else v / 1_000_000
    return Price(
        input_cost_per_token=input_per_m / 1_000_000,
        output_cost_per_token=output_per_m / 1_000_000,
        input_cost_per_audio_token=per_token(audio_per_m),
        cache_read_input_token_cost=per_token(cache_per_m),
        output_cost_per_reasoning_token=per_token(reasoning_per_m),
    )


@pytest.fixture(autouse=True)
def _empty_user_cache(_test_price_cache):
    """Start each pricing test with an empty cache.

    The repo-level ``conftest`` seeds a populated cache so CLI/inline
    tests can assert on priced totals. Pricing tests, in contrast,
    exercise the cache-population/absence paths directly, so they
    want a blank slate. We wipe the file the conftest wrote and let
    individual tests re-populate via ``refresh_prices`` or direct
    write.
    """
    if _test_price_cache.exists():
        _test_price_cache.unlink()
    default_prices.cache_clear()
    yield _test_price_cache
    default_prices.cache_clear()


def test_default_prices_empty_without_cache():
    """No cache, no bundle → no prices. Users run refresh-prices to populate."""
    assert default_prices() == {}


def test_price_cost_multiplies_per_token():
    # $2.50 per 1M input, $15 per 1M output → 1M input = $2.50 exactly.
    p = _m(2.5, 15.0)
    assert p.cost(1_000_000, 0) == pytest.approx(2.5)
    assert p.cost(0, 1_000_000) == pytest.approx(15.0)
    assert p.cost(500_000, 200_000) == pytest.approx(2.5 * 0.5 + 15.0 * 0.2)


def test_cost_for_prices_audio_at_audio_rate():
    # Gemini 2.5 flash shape: text $0.30/M, audio $1.00/M.
    p = _rich(0.3, 2.5, audio_per_m=1.0)
    breakdown = TokenBreakdown(
        input_text=100_000, input_audio=1_000_000, input_cached=0,
        output=0, output_reasoning=0,
    )
    # text: 100k * $0.30/M = $0.03; audio: 1M * $1.00/M = $1.00
    assert p.cost_for(breakdown) == pytest.approx(0.03 + 1.0)


def test_cost_for_audio_falls_back_to_text_rate_when_unset():
    # Gemini 3.x Pro: LiteLLM doesn't publish a separate audio rate, so
    # audio should price at the text rate without a warning.
    p = _rich(2.0, 12.0)  # no audio rate
    breakdown = TokenBreakdown(
        input_text=0, input_audio=500_000, input_cached=0,
        output=0, output_reasoning=0,
    )
    assert p.cost_for(breakdown) == pytest.approx(1.0)  # 500k * $2/M


def test_cost_for_cache_read_at_discounted_rate():
    p = _rich(0.3, 2.5, cache_per_m=0.03)  # cache is 10× cheaper
    breakdown = TokenBreakdown(
        input_text=0, input_audio=0, input_cached=1_000_000,
        output=0, output_reasoning=0,
    )
    assert p.cost_for(breakdown) == pytest.approx(0.03)


def test_cost_for_reasoning_falls_back_to_output_rate():
    p = _rich(0.3, 2.5)  # no distinct reasoning rate
    breakdown = TokenBreakdown(
        input_text=0, input_audio=0, input_cached=0,
        output=1000, output_reasoning=2000,
    )
    # 3k total output at $2.50/M
    assert p.cost_for(breakdown) == pytest.approx(3_000 * 2.5 / 1_000_000)


def test_cost_is_cost_for_text_only_equivalent():
    """The legacy cost(input, output) API is a text-only wrapper."""
    p = _rich(0.3, 2.5, audio_per_m=1.0, cache_per_m=0.03)
    # Same numbers going through both APIs — the breakdown-free call
    # must produce the same result as an explicit text-only breakdown.
    via_cost = p.cost(1_000_000, 500_000)
    via_cost_for = p.cost_for(TokenBreakdown.text_only(1_000_000, 500_000))
    assert via_cost == via_cost_for


def test_load_prices_reads_optional_audio_and_cache_fields(tmp_path: Path):
    path = tmp_path / "p.yaml"
    path.write_text(
        json.dumps(
            {
                "gemini-2.5-flash": {
                    "input_cost_per_token": 3e-7,
                    "output_cost_per_token": 2.5e-6,
                    "input_cost_per_audio_token": 1e-6,
                    "cache_read_input_token_cost": 3e-8,
                    "output_cost_per_reasoning_token": 2.5e-6,
                },
                "plain-model": {
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 5e-6,
                },
            }
        )
    )
    table = load_prices(path)
    rich = table["gemini-2.5-flash"]
    assert rich.input_cost_per_audio_token == pytest.approx(1e-6)
    assert rich.cache_read_input_token_cost == pytest.approx(3e-8)
    assert rich.output_cost_per_reasoning_token == pytest.approx(2.5e-6)
    # Plain model keeps the optional fields as None.
    plain = table["plain-model"]
    assert plain.input_cost_per_audio_token is None
    assert plain.cache_read_input_token_cost is None
    assert plain.output_cost_per_reasoning_token is None


def test_canonical_strips_provider_prefix_and_variants():
    assert _canonical("gemini/gemini-3-flash-preview") == "gemini-3-flash-preview"
    assert _canonical("anthropic/claude-opus-4-6") == "claude-opus-4-6"
    assert _canonical("openrouter/gpt-5.4-mini") == "gpt-5.4-mini"
    assert _canonical("gemini/gemini-3.1-pro-preview-customtools") == "gemini-3.1-pro-preview"
    assert _canonical("gemini/gemini-flash-latest") == "gemini-flash"
    assert _canonical("anthropic/claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert _canonical("mistral/mistral-tiny") == "mistral-tiny"
    assert _canonical("mistral/mistral-small-latest") == "mistral-small"


def test_resolve_prefers_resolved_model_when_raw_is_ambiguous():
    """Raw name may be an llm alias; prefer the resolver's provider-resolved name."""
    table = {
        "gemini-flash": _m(0.3, 2.5),
        "gemini-3-flash-preview": _m(0.5, 3.0),
    }
    price = resolve(
        "gemini/gemini-flash-latest", "gemini-3-flash-preview", table=table
    )
    assert price == _m(0.5, 3.0)


def test_resolve_falls_back_to_raw_when_resolved_missing():
    table = {"claude-opus-4-6": _m(5.0, 25.0)}
    price = resolve("anthropic/claude-opus-4-6", None, table=table)
    assert price == _m(5.0, 25.0)


def test_resolve_returns_none_for_unknown():
    assert resolve("totally-made-up-model", table={}) is None


def test_load_prices_flat_litellm_shape(tmp_path: Path):
    """The loader reads the LiteLLM field names straight off disk."""
    path = tmp_path / "p.yaml"
    path.write_text(
        """
        my-model:
          input_cost_per_token: 1.5e-6
          output_cost_per_token: 7e-6
        partial-model:
          input_cost_per_token: 2e-6
        """
    )
    table = load_prices(path)
    # Partial entries (missing one side of the pair) are skipped.
    assert table == {"my-model": _m(1.5, 7.0)}


def test_load_prices_reads_litellm_json_verbatim(tmp_path: Path):
    """yaml.safe_load handles JSON, so a LiteLLM file works as-is."""
    path = tmp_path / "litellm.json"
    path.write_text(
        json.dumps(
            {
                "sample_spec": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
                "gpt-4o": {
                    "input_cost_per_token": 2.5e-6,
                    "output_cost_per_token": 1e-5,
                    "litellm_provider": "openai",
                },
            }
        )
    )
    table = load_prices(path)
    # sample_spec is skipped; extra metadata fields are ignored.
    assert "sample_spec" not in table
    assert table["gpt-4o"] == _m(2.5, 10.0)


def test_resolve_honours_custom_table():
    custom = {"my-model": _m(99.0, 101.0)}
    assert resolve("my-model", None, table=custom) == _m(99.0, 101.0)


def test_refresh_prices_writes_user_cache(_empty_user_cache, monkeypatch):
    """`refresh_prices` downloads the catalog and invalidates the cache."""
    cache_path = _empty_user_cache
    payload = json.dumps(
        {
            "gpt-4o": {
                "input_cost_per_token": 2.5e-6,
                "output_cost_per_token": 1e-5,
            }
        }
    ).encode()

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(pricing, "urlopen", lambda req, timeout=30.0: _FakeResp())

    dest = refresh_prices()
    assert dest == cache_path
    assert cache_path.read_bytes() == payload
    # After refresh the loader sees the downloaded entry.
    assert load_prices(cache_path) == {"gpt-4o": _m(2.5, 10.0)}


def test_default_prices_reads_user_cache(_empty_user_cache):
    """Once a cache file exists, default_prices() uses it automatically."""
    cache_path = _empty_user_cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "claude-opus-4-6": {
                    "input_cost_per_token": 5e-6,
                    "output_cost_per_token": 2.5e-5,
                }
            }
        )
    )
    default_prices.cache_clear()
    table = default_prices()
    assert table == {"claude-opus-4-6": _m(5.0, 25.0)}


def test_refresh_prices_rejects_invalid_json(_empty_user_cache, monkeypatch):
    """A bad download must not clobber an existing good cache."""
    cache_path = _empty_user_cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"keep": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}}
        )
    )
    original = cache_path.read_bytes()

    class _BadResp:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return b"<!DOCTYPE html><html>rate limited</html>"

    monkeypatch.setattr(pricing, "urlopen", lambda req, timeout=30.0: _BadResp())

    with pytest.raises(json.JSONDecodeError):
        refresh_prices()
    # Previous cache intact.
    assert cache_path.read_bytes() == original
