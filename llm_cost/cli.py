"""``llm cost`` Click command group."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import llm
import sqlite_utils

from .pricing import default_prices, load_prices
from .summary import Summary, local_day_bounds, models_without_prices, summarise


def _logs_db_path() -> Path:
    return llm.user_dir() / "logs.db"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _today_local() -> date:
    return datetime.now().astimezone().date()


def _shared_options(func: Callable) -> Callable:
    """Options common to every reporting subcommand."""
    func = click.option(
        "--model",
        "model_glob",
        type=str,
        help="SQL LIKE pattern over model (e.g. 'gemini/%').",
    )(func)
    func = click.option(
        "--prices",
        "prices_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Path to a YAML price table (overrides the bundled one).",
    )(func)
    func = click.option(
        "--json",
        "as_json",
        is_flag=True,
        default=False,
        help="Emit machine-readable JSON instead of a table.",
    )(func)
    func = click.option(
        "--db",
        "db_path",
        type=click.Path(path_type=Path),
        default=None,
        help="Path to logs.db (defaults to llm's configured location).",
    )(func)
    return func


def _load_price_table(prices_path: Path | None) -> dict | None:
    if prices_path is not None:
        return load_prices(prices_path)
    env_path = os.environ.get("LLM_COST_PRICES")
    if env_path:
        return load_prices(env_path)
    return None


def _report(
    since,
    until,
    label: str,
    model_glob: str | None,
    prices_path: Path | None,
    as_json: bool,
    db_path: Path | None,
) -> None:
    prices = _load_price_table(prices_path)
    db = sqlite_utils.Database(str(db_path or _logs_db_path()))
    summary = summarise(db, since=since, until=until, model_glob=model_glob, prices=prices)
    if as_json:
        click.echo(_render_json(summary, label))
    else:
        click.echo(_render_table(summary, label))


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _render_table(summary: Summary, label: str) -> str:
    if not summary.rows:
        return f"No responses logged {label}."

    headers = ("model", "resps", "in", "out", "cost (USD)", "source")
    rows = []
    for row in summary.rows:
        display_model = row.resolved_model or row.model
        if row.logged_cost_usd > 0:
            source = "logged"
        elif row.priced:
            source = "priced"
        else:
            source = "unpriced"
        rows.append(
            (
                display_model,
                str(row.response_count),
                _format_tokens(row.input_tokens),
                _format_tokens(row.output_tokens),
                f"${row.best_cost_usd:,.4f}",
                source,
            )
        )

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=False))

    out = [
        f"Spend {label}",
        line(headers),
        line(["-" * w for w in widths]),
    ]
    out.extend(line(r) for r in rows)
    out.append(line(["-" * w for w in widths]))
    out.append(
        line(
            [
                "TOTAL",
                str(summary.total_responses),
                _format_tokens(summary.total_input),
                _format_tokens(summary.total_output),
                f"${summary.total_cost_usd:,.4f}",
                "",
            ]
        )
    )
    if summary.has_unpriced:
        unpriced = ", ".join(models_without_prices(summary))
        out.append("")
        out.append(f"Note: no price for: {unpriced}")
        out.append(
            "      Override with --prices PATH or set LLM_COST_PRICES; "
            "tokens are counted but cost shown as $0."
        )
    return "\n".join(out)


def _render_json(summary: Summary, label: str) -> str:
    payload = {
        "label": label,
        "since_utc": summary.since_utc.isoformat() if summary.since_utc else None,
        "until_utc": summary.until_utc.isoformat() if summary.until_utc else None,
        "total": {
            "responses": summary.total_responses,
            "input_tokens": summary.total_input,
            "output_tokens": summary.total_output,
            "cost_usd": summary.total_cost_usd,
        },
        "rows": [
            {
                "model": r.model,
                "resolved_model": r.resolved_model,
                "responses": r.response_count,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "logged_cost_usd": r.logged_cost_usd,
                "priced_cost_usd": r.priced_cost_usd,
                "best_cost_usd": r.best_cost_usd,
                "priced": r.priced,
            }
            for r in summary.rows
        ],
        "unpriced_models": list(models_without_prices(summary)),
    }
    return json.dumps(payload, indent=2)


def register_commands(cli: click.Group) -> None:
    @cli.group(name="cost", invoke_without_command=True)
    @click.option("--since", type=str, help="Start date YYYY-MM-DD (inclusive, local tz).")
    @click.option("--until", type=str, help="End date YYYY-MM-DD (inclusive, local tz).")
    @click.option("--days", type=int, help="Last N days (including today).")
    @_shared_options
    @click.pass_context
    def cost_group(
        ctx: click.Context,
        since: str | None,
        until: str | None,
        days: int | None,
        model_glob: str | None,
        prices_path: Path | None,
        as_json: bool,
        db_path: Path | None,
    ) -> None:
        """Report token usage and spend from the llm logs database."""
        if ctx.invoked_subcommand is not None:
            return

        if days is not None:
            today = _today_local()
            start, _ = local_day_bounds(today - timedelta(days=days - 1))
            _, end = local_day_bounds(today)
            label = f"last {days} day{'s' if days != 1 else ''}"
        elif since or until:
            start = local_day_bounds(_parse_date(since))[0] if since else None
            end = local_day_bounds(_parse_date(until))[1] if until else None
            parts = []
            if since:
                parts.append(f"since {since}")
            if until:
                parts.append(f"until {until}")
            label = " ".join(parts)
        else:
            start = end = None
            label = "all time"

        _report(start, end, label, model_glob, prices_path, as_json, db_path)

    @cost_group.command(name="today")
    @_shared_options
    def cost_today(
        model_glob: str | None,
        prices_path: Path | None,
        as_json: bool,
        db_path: Path | None,
    ) -> None:
        """Spend today (local time)."""
        start, end = local_day_bounds(_today_local())
        _report(start, end, "today", model_glob, prices_path, as_json, db_path)

    @cost_group.command(name="yesterday")
    @_shared_options
    def cost_yesterday(
        model_glob: str | None,
        prices_path: Path | None,
        as_json: bool,
        db_path: Path | None,
    ) -> None:
        """Spend yesterday (local time)."""
        start, end = local_day_bounds(_today_local() - timedelta(days=1))
        _report(start, end, "yesterday", model_glob, prices_path, as_json, db_path)

    @cost_group.command(name="models")
    @click.option(
        "--prices",
        "prices_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Path to a YAML price table (overrides the bundled one).",
    )
    def cost_models_cmd(prices_path: Path | None) -> None:
        """List models with known prices in the active price table."""
        table = load_prices(prices_path) if prices_path else default_prices()
        if not table:
            click.echo("No prices loaded.")
            return
        width = max(len(k) for k in table)
        click.echo(f"{'model'.ljust(width)}  input $/1M  output $/1M")
        for name in sorted(table):
            p = table[name]
            click.echo(
                f"{name.ljust(width)}  {p.input_per_mtok:>10.3f}  {p.output_per_mtok:>11.3f}"
            )
