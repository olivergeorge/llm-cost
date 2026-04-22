"""llm-cost: token and spend reports over the llm logs database."""

from __future__ import annotations

from typing import Any

import click
from llm import hookimpl

from . import inline
from .cli import register_commands as _register_cli_commands

__all__ = ["register_commands", "after_log_to_db"]


def _cost_flag_callback(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
    if value:
        inline.enable()
    return value


def _no_cost_flag_callback(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
    """Explicit off-switch so ``LLM_COST=1`` in the environment can be
    overridden for a single invocation without unsetting the var."""
    if value:
        inline.disable()
    return value


@hookimpl
def register_commands(cli: click.Group) -> None:
    """Register ``llm cost`` subcommands and inject ``--cost`` into prompt/chat."""
    _register_cli_commands(cli)

    # Attach --cost / --no-cost to the existing prompt/chat commands,
    # mirroring the llm-replay flag-injection pattern. Idempotent: skip
    # if a flag is already present (tests may re-register).
    flag_specs = (
        (
            "--cost",
            _cost_flag_callback,
            "Show the cost of the response to stderr (llm-cost)",
        ),
        (
            "--no-cost",
            _no_cost_flag_callback,
            "Override LLM_COST=1 for this invocation (llm-cost)",
        ),
    )
    for cmd_name in ("prompt", "chat"):
        cmd = cli.commands.get(cmd_name)
        if cmd is None:
            continue
        existing = {opt for p in cmd.params for opt in (p.opts or ())}
        for flag, callback, help_text in flag_specs:
            if flag in existing:
                continue
            cmd.params.append(
                click.Option(
                    [flag],
                    is_flag=True,
                    default=False,
                    expose_value=False,
                    callback=callback,
                    help=help_text,
                )
            )


@hookimpl
def after_log_to_db(response: Any, db: Any) -> None:
    """Emit the inline cost line when ``--cost`` / ``LLM_COST`` is set.

    Fires at the tail of ``_BaseResponse.log_to_db`` — i.e. immediately
    after llm core has printed its ``Token usage:`` line (when ``-u``
    is passed) and persisted the response row. ``db`` is unused here
    because we compute cost from the response's in-memory token counts
    using our price table, avoiding an extra round-trip.
    """
    if not inline.is_enabled():
        return
    inline.emit_cost_for_response(response)
