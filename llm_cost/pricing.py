"""Price table for common llm models.

Prices are USD per **one million** tokens. The bundled ``prices.yaml``
tracks the snapshot kept in ``tasict/acidifier/tool/models.json`` at
plugin-release time. Point ``--prices PATH`` or ``LLM_COST_PRICES=PATH``
at your own YAML to override — same schema:

    model-name:
      input: 2.5     # USD / 1M input tokens
      output: 15.0   # USD / 1M output tokens

The resolver normalises common prefix/suffix variants (``gemini/`` /
``anthropic/``, trailing ``-latest`` or ``-customtools``, date suffixes
like ``-20251001``) so that both the raw ``model`` column and the
``resolved_model`` column from llm's logs can be looked up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok / 1_000_000
            + output_tokens * self.output_per_mtok / 1_000_000
        )


_BUNDLED_PRICES_PATH = Path(__file__).parent / "prices.yaml"

_PROVIDER_PREFIXES = (
    "gemini/",
    "openrouter/",
    "anthropic/",
    "openai/",
    "google/",
)
_VARIANT_SUFFIXES = ("-customtools", "-thinking", "-latest")
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def _canonical(name: str) -> str:
    n = name.lower()
    for prefix in _PROVIDER_PREFIXES:
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    n = _DATE_SUFFIX_RE.sub("", n)
    for suffix in _VARIANT_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n


def load_prices(path: Path | str) -> dict[str, Price]:
    """Load a price YAML.

    Two shapes are accepted:

    1. Flat mapping (the shape of ``prices.yaml``)::

           claude-opus-4-6:
             input: 5.0
             output: 25.0

    2. ``models.json``-style wrapper::

           models:
             anthropic/claude-opus-4-6:
               input_cost_per_1m: 5.0
               output_cost_per_1m: 25.0

    Entries missing either price are skipped so that files with partial
    data (e.g. free local models) don't raise — those models simply fall
    back to the default table.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    if isinstance(data, dict) and "models" in data and isinstance(data["models"], dict):
        models = data["models"]
    else:
        models = data

    out: dict[str, Price] = {}
    for name, spec in models.items():
        if not isinstance(spec, dict):
            continue
        inp = spec.get("input", spec.get("input_cost_per_1m"))
        outp = spec.get("output", spec.get("output_cost_per_1m"))
        if inp is None or outp is None:
            continue
        out[_canonical(name)] = Price(float(inp), float(outp))
    return out


@lru_cache(maxsize=1)
def default_prices() -> dict[str, Price]:
    """Load and cache the bundled price table."""
    return load_prices(_BUNDLED_PRICES_PATH)


def resolve(
    model: str,
    resolved_model: str | None = None,
    table: dict[str, Price] | None = None,
) -> Price | None:
    """Look up a price, preferring the provider-resolved name when given.

    Returns None when neither key is known — callers should surface the
    model in an "unpriced" bucket rather than silently zero-cost it.
    """
    prices = table if table is not None else default_prices()
    for candidate in (resolved_model, model):
        if not candidate:
            continue
        key = _canonical(candidate)
        if key in prices:
            return prices[key]
    return None
