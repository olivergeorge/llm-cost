"""Price table for common llm models.

The schema accepted by ``--prices PATH`` / ``LLM_COST_PRICES=PATH`` and
the file written by ``llm cost refresh-prices`` all follow LiteLLM's
``model_prices_and_context_window.json`` — a **flat** mapping of model
name to per-token costs:

    claude-opus-4-6:
      input_cost_per_token: 5e-6     # USD per input token
      output_cost_per_token: 2.5e-5  # USD per output token

An entry can be pasted straight from LiteLLM's catalog and work
unchanged. The canonicaliser normalises provider prefixes (``gemini/``,
``anthropic/``, ...), trailing ``-latest`` / ``-customtools``, and
date suffixes like ``-20251001`` so that both ``model`` and
``resolved_model`` from llm's logs resolve against the same entry.

**No price data is bundled with the plugin** — pricing drifts fast and
a stale snapshot is worse than none. On first install every model
shows as "unpriced" (tokens still counted, cost shown as $0). Run
``llm cost refresh-prices`` once to download the LiteLLM catalog; it
writes to :func:`user_cache_path` and is picked up automatically.

The active price table is resolved in this order:

1. Explicit ``--prices PATH`` CLI flag (callers pass through
   :func:`load_prices`).
2. ``LLM_COST_PRICES`` environment variable (same).
3. User cache at :func:`user_cache_path` — populated by
   ``llm cost refresh-prices``.
4. Empty table (all models "unpriced") if none of the above exist.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

import yaml


@dataclass(frozen=True)
class Price:
    """Per-token USD cost for a model.

    Stored per-token to match LiteLLM's schema. Multiply by token counts
    directly in :meth:`cost`; no divide-by-a-million dance at the call
    site.
    """

    input_cost_per_token: float
    output_cost_per_token: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_token
            + output_tokens * self.output_cost_per_token
        )


_PROVIDER_PREFIXES = (
    "gemini/",
    "openrouter/",
    "anthropic/",
    "openai/",
    "google/",
    "mistral/",
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
    """Load a LiteLLM-style price YAML.

    Flat mapping only — one top-level key per model, each entry carrying
    ``input_cost_per_token`` and ``output_cost_per_token``. Entries
    missing either field are skipped so a partial table (e.g. free
    local models without a full pricing block) doesn't raise; those
    models fall back to the default table.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        return {}

    out: dict[str, Price] = {}
    for name, spec in data.items():
        # LiteLLM's catalog leads with a ``sample_spec`` documentation
        # entry whose cost fields are placeholders — skip it rather than
        # surface a phantom model.
        if name == "sample_spec":
            continue
        if not isinstance(spec, dict):
            continue
        inp = spec.get("input_cost_per_token")
        outp = spec.get("output_cost_per_token")
        if inp is None or outp is None:
            continue
        out[_canonical(name)] = Price(float(inp), float(outp))
    return out


# LiteLLM's canonical catalog. Pinning to ``main`` is deliberate — the
# catalog doesn't tag releases, and users invoking
# ``llm cost refresh-prices`` expect "latest".
LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)


def user_cache_path() -> Path:
    """Where ``llm cost refresh-prices`` writes its snapshot.

    Uses ``click.get_app_dir`` for platform-correct placement
    (``~/.config/llm-cost/`` on Linux, ``~/Library/Application Support/``
    on macOS, ``%APPDATA%\\llm-cost\\`` on Windows). ``click`` is a
    transitive dependency via ``llm`` itself, so this doesn't add a
    runtime requirement.
    """
    import click

    return Path(click.get_app_dir("llm-cost")) / "prices.json"


def refresh_prices(
    url: str = LITELLM_PRICES_URL,
    dest: Path | None = None,
    *,
    timeout: float = 30.0,
) -> Path:
    """Download LiteLLM's price catalog to the user cache and return the path.

    The written file is consumed verbatim by :func:`load_prices` — the
    catalog already ships as a flat mapping of model → cost fields,
    which is the schema this loader expects. ``default_prices()`` picks
    it up automatically on the next call; callers who hold a cached
    table should invalidate via ``default_prices.cache_clear()``.
    """
    target = dest if dest is not None else user_cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # GitHub's raw hosting sometimes 403s unless a UA is set.
    request = Request(url, headers={"User-Agent": "llm-cost"})
    with urlopen(request, timeout=timeout) as resp:
        body = resp.read()
    # Validate JSON before overwriting, so a bad response doesn't
    # clobber a previously-good cache.
    json.loads(body)
    target.write_bytes(body)
    default_prices.cache_clear()
    return target


@lru_cache(maxsize=1)
def default_prices() -> dict[str, Price]:
    """Return the active default price table.

    Reads the user cache populated by ``llm cost refresh-prices``.
    Returns an empty table when the cache is absent — no pricing data
    is bundled with the package, so every model surfaces as "unpriced"
    until the user runs the refresh command. That's a deliberate
    trade-off: pricing drifts fast and a stale snapshot is worse than
    an explicit prompt to refresh.
    """
    cache = user_cache_path()
    if cache.exists():
        return load_prices(cache)
    return {}


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
