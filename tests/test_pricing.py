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
    assert _canonical("mistral/mistral-tiny") == "mistral-tiny"
    assert _canonical("mistral/mistral-small-latest") == "mistral-small"


def test_bundled_prices_cover_newly_added_models():
    """Sanity-check a representative slice of the price table additions."""
    table = default_prices()
    assert resolve("gpt-5", table=table) == Price(1.25, 10.0)
    assert resolve("gpt-5-nano", table=table) == Price(0.05, 0.4)
    assert resolve("gpt-4o", table=table) == Price(2.5, 10.0)
    assert resolve("o3-pro", table=table) == Price(20.0, 80.0)
    assert resolve("codex-mini", table=table) == Price(1.5, 6.0)
    assert resolve("anthropic/claude-opus-4-5-20251101", table=table) == Price(5.0, 25.0)
    assert resolve("anthropic/claude-sonnet-4-5", table=table) == Price(3.0, 15.0)
    assert resolve("anthropic/claude-opus-4-0", table=table) == Price(15.0, 75.0)
    # Preview snapshots that don't match the 8-digit date regex get explicit entries.
    assert resolve("gemini/gemini-2.5-flash-preview-05-20", table=table) == Price(0.3, 2.5)
    assert resolve("gemini/gemini-2.5-flash-lite-preview-09-2025", table=table) == Price(0.1, 0.4)
    assert resolve("gemini/gemini-2.5-pro-preview-06-05", table=table) == Price(1.25, 10.0)
    # -latest alias falls through to the bare canonical.
    assert resolve("gemini/gemini-flash-latest", table=table) == Price(0.3, 2.5)
    assert resolve("gemini/gemini-flash-lite-latest", table=table) == Price(0.1, 0.4)
    # Mistral picks up via the new provider prefix.
    assert resolve("mistral/mistral-tiny", table=table) == Price(0.25, 0.25)
    assert resolve("mistral/devstral-small", table=table) == Price(0.1, 0.3)
    # Local / free models are priced explicitly at $0.
    assert resolve("gemma4:26b", table=table) == Price(0.0, 0.0)


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
