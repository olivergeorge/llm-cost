"""Inline per-response cost display — the `--cost` flag analog of `-u`.

When enabled (via ``--cost`` or ``LLM_COST=1``), every logged response
gets a ``Cost: $0.0017 (priced)`` line printed to stderr alongside
llm core's ``Token usage:`` line. The hook fires from
``after_log_to_db`` so the order is predictable and the cost reflects
the final token counts llm persisted.

The amount is rendered via :func:`llm_cost.cli.format_money`, which
drops to four decimals for sub-cent amounts — otherwise a short prompt
on a cheap model would display as ``$0.00`` and look free when it
isn't.

Enable state follows the same ContextVar-plus-env pattern as
llm-replay so concurrent async library use doesn't leak enablement.
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar

import click

from . import pricing
from .cli import format_money
from .summary import canonical_key, parse_token_details

_ENABLED: ContextVar[bool | None] = ContextVar("llm_cost_inline_enabled", default=None)


def _env_default() -> bool:
    return bool(os.environ.get("LLM_COST"))


def is_enabled() -> bool:
    value = _ENABLED.get()
    if value is None:
        return _env_default()
    return value


def enable() -> None:
    _ENABLED.set(True)


def disable() -> None:
    _ENABLED.set(False)


def _alias_map() -> dict[str, str]:
    try:
        import llm

        return {name: m.model_id for name, m in llm.get_model_aliases().items()}
    except Exception:  # pragma: no cover - defensive
        return {}


def format_cost_line(
    model_id: str,
    resolved_model: str | None,
    input_tokens: int,
    output_tokens: int,
    alias_map: dict[str, str] | None = None,
    prices: dict[str, pricing.Price] | None = None,
    token_details: str | None = None,
) -> str:
    """Compute the cost and render the one-line display.

    Separated from the hook for testability — callers pass in the
    alias map / price table explicitly. ``token_details`` (raw JSON from
    the provider) lets us split the input into text/audio/cached so
    Gemini audio calls don't under-price by 3–7×.
    """
    table = prices if prices is not None else pricing.default_prices()
    amap = alias_map if alias_map is not None else _alias_map()
    key = canonical_key(model_id, resolved_model, amap)
    price = pricing.resolve(key, None, table)
    if price is not None:
        breakdown = parse_token_details(token_details, input_tokens, output_tokens)
        cost = price.cost_for(breakdown)
        source = "priced"
    else:
        cost = 0.0
        source = "unpriced"
    return f"Cost: {format_money(cost)} ({source})"


def emit_cost_for_response(response) -> None:
    """Print the cost line to stderr for a single llm Response.

    Invoked from the ``after_log_to_db`` hook. Tolerates missing fields
    so library callers that don't populate token counts don't crash —
    we just show ``$0.0000 (unpriced)``.
    """
    try:
        model_id = getattr(response.prompt.model, "model_id", None) or ""
    except AttributeError:
        model_id = ""
    resolved = getattr(response, "resolved_model", None)
    inp = getattr(response, "input_tokens", None) or 0
    outp = getattr(response, "output_tokens", None) or 0
    # Response.token_details is a dict on the in-memory object; the
    # parser expects the JSON string shape stored in sqlite. Round-trip
    # so the same helper works in both paths.
    td_obj = getattr(response, "token_details", None)
    token_details = json.dumps(td_obj) if td_obj else None

    line = format_cost_line(
        model_id, resolved, int(inp), int(outp), token_details=token_details
    )
    click.echo(click.style(line, fg="yellow", bold=True), err=True)
