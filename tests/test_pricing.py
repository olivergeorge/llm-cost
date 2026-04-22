from __future__ import annotations

from pathlib import Path

from llm_cost.pricing import Price, _canonical, default_prices, load_prices, resolve


def test_bundled_prices_load():
    table = default_prices()
    assert "claude-opus-4-6" in table
    assert table["claude-opus-4-6"] == Price(5.0, 25.0)


def test_price_cost_round_trip():
    p = Price(2.5, 15.0)
    assert p.cost(1_000_000, 0) == 2.5
    assert p.cost(0, 1_000_000) == 15.0
    # Mixed, fractional tokens
    assert p.cost(500_000, 200_000) == 2.5 * 0.5 + 15.0 * 0.2


def test_canonical_strips_provider_prefix_and_variants():
    assert _canonical("gemini/gemini-3-flash-preview") == "gemini-3-flash-preview"
    assert _canonical("anthropic/claude-opus-4-6") == "claude-opus-4-6"
    assert _canonical("openrouter/gpt-5.4-mini") == "gpt-5.4-mini"
    assert _canonical("gemini/gemini-3.1-pro-preview-customtools") == "gemini-3.1-pro-preview"
    assert _canonical("gemini/gemini-flash-latest") == "gemini-flash"
    assert _canonical("anthropic/claude-haiku-4-5-20251001") == "claude-haiku-4-5"


def test_resolve_prefers_resolved_model_when_raw_is_ambiguous():
    # Raw name is an alias llm uses internally; resolved is the real model.
    price = resolve("gemini/gemini-flash-latest", "gemini-3-flash-preview")
    assert price == Price(0.5, 3.0)


def test_resolve_falls_back_to_raw_when_resolved_missing():
    price = resolve("anthropic/claude-opus-4-6", None)
    assert price == Price(5.0, 25.0)


def test_resolve_returns_none_for_unknown():
    assert resolve("totally-made-up-model") is None


def test_load_prices_flat_shape(tmp_path: Path):
    path = tmp_path / "p.yaml"
    path.write_text(
        """
        my-model:
          input: 1.5
          output: 7.0
        partial-model:
          input: 2.0
        """
    )
    table = load_prices(path)
    assert table == {"my-model": Price(1.5, 7.0)}


def test_load_prices_wrapped_shape(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text(
        """
        models:
          anthropic/claude-opus-4-6:
            input_cost_per_1m: 5.0
            output_cost_per_1m: 25.0
          gemma4:26b:
            input_cost_per_1m: 0
            output_cost_per_1m: 0
        """
    )
    table = load_prices(path)
    assert table["claude-opus-4-6"] == Price(5.0, 25.0)
    assert table["gemma4:26b"] == Price(0.0, 0.0)


def test_resolve_honours_custom_table():
    custom = {"my-model": Price(99.0, 101.0)}
    assert resolve("my-model", None, table=custom) == Price(99.0, 101.0)
    # Default-table hit doesn't leak when a custom table is passed
    assert resolve("claude-opus-4-6", None, table=custom) is None
