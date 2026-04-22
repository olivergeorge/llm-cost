"""``llm cost`` Click command group."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import click
import llm
import sqlite_utils

from .pricing import default_prices, load_prices
from .summary import (
    DailyRow,
    ExpensiveResponse,
    Headlines,
    Summary,
    daily_summary,
    headlines,
    local_day_bounds,
    models_without_prices,
    summarise,
    top_responses,
)


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


def _alias_map() -> dict[str, str]:
    """Snapshot of llm's alias → canonical ``model_id`` registry.

    Used to collapse historical variants (e.g. ``gemini-3-flash-preview``,
    ``gemini/gemini-3-flash-preview``, and the ``gemini-flash-latest``
    alias) into a single grouped row.
    """
    try:
        return {name: m.model_id for name, m in llm.get_model_aliases().items()}
    except Exception:  # pragma: no cover - defensive against llm API drift
        return {}


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
    summary = summarise(
        db,
        since=since,
        until=until,
        model_glob=model_glob,
        prices=prices,
        alias_map=_alias_map(),
    )
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
        rows.append(
            (
                row.model,
                str(row.response_count),
                _format_tokens(row.input_tokens),
                _format_tokens(row.output_tokens),
                f"${row.best_cost_usd:,.4f}",
                row.source,
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


def _bar(value: float, max_value: float, width: int = 20) -> str:
    if max_value <= 0:
        return ""
    fill = int(round((value / max_value) * width))
    return "▓" * fill


def _render_daily(days: tuple[DailyRow, ...], head: Headlines) -> str:
    out: list[str] = []
    if not days:
        out.append("No responses logged yet. Run `llm prompt ...` to get started.")
        return "\n".join(out)

    max_cost = max((d.cost_usd for d in days), default=0.0)
    bar_width = 20
    date_w = 10
    resps_w = max(len("resps"), *(len(str(d.responses)) for d in days))
    cost_strs = [f"${d.cost_usd:,.4f}" for d in days]
    cost_w = max(len("cost"), *(len(s) for s in cost_strs))

    out.append(f"Spend — last {len(days)} days")
    out.append("")
    out.append(
        f"{'date'.ljust(date_w)}  {'resps'.rjust(resps_w)}  "
        f"{'cost'.rjust(cost_w)}  bar"
    )
    out.append(
        f"{'-' * date_w}  {'-' * resps_w}  {'-' * cost_w}  {'-' * bar_width}"
    )
    for row, cost_str in zip(days, cost_strs, strict=True):
        out.append(
            f"{row.day.isoformat().ljust(date_w)}  "
            f"{str(row.responses).rjust(resps_w)}  "
            f"{cost_str.rjust(cost_w)}  "
            f"{_bar(row.cost_usd, max_cost, bar_width)}"
        )

    out.append("")
    label_w = len("This month")
    out.append(f"{'Today'.ljust(label_w)}  ${head.today:,.4f}")
    out.append(f"{'This week'.ljust(label_w)}  ${head.this_week:,.4f}  (last 7 days)")
    out.append(f"{'This month'.ljust(label_w)}  ${head.this_month:,.4f}  (month-to-date)")
    out.append(f"{'All time'.ljust(label_w)}  ${head.all_time:,.4f}")

    if head.top_models_month:
        out.append("")
        out.append("Top models this month:")
        # Scale against the month total, not the top-N subtotal, so the
        # percentage shown is the true share of spend.
        month_total = head.this_month or 1.0
        model_w = max(len(r.model) for r in head.top_models_month)
        for r in head.top_models_month:
            pct = r.best_cost_usd / month_total * 100
            out.append(f"  {r.model.ljust(model_w)}  ${r.best_cost_usd:>10,.4f}  ({pct:>4.1f}%)")

    out.append("")
    out.append(
        "Drill down: `llm cost today` · `llm cost --since YYYY-MM-DD` · `llm cost all`"
    )
    return "\n".join(out)


def _local_clock(dt_iso: str) -> str:
    """Parse an ISO UTC string and render as local ``YYYY-MM-DD HH:MM``."""
    try:
        dt = datetime.fromisoformat(dt_iso)
    except ValueError:
        return dt_iso[:16]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def _render_top(rows: tuple[ExpensiveResponse, ...], by: str) -> str:
    if not rows:
        return "No responses in that window."

    headers = ("datetime (local)", "model", "in", "out", "cost", "prompt")
    table_rows = []
    for r in rows:
        table_rows.append(
            (
                _local_clock(r.datetime_utc),
                r.model,
                _format_tokens(r.input_tokens),
                _format_tokens(r.output_tokens),
                f"${r.cost_usd:,.4f}",
                r.prompt_preview,
            )
        )

    # The prompt column stays on the right and isn't padded — it's the tail
    # of the line so ragged endings are fine.
    static_cols = 5
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(static_cols)
    ]

    def line(cells):
        head_cells = [c.ljust(w) for c, w in zip(cells[:static_cols], widths, strict=True)]
        return "  ".join(head_cells + [cells[static_cols]])

    out = [
        f"Top {len(rows)} expensive requests (by {by})",
        "",
        line(headers),
        line(["-" * w for w in widths] + ["-" * 40]),
    ]
    out.extend(line(r) for r in table_rows)
    out.append("")
    out.append(
        "Tip: install llm-confirm-tokens to catch big prompts before they send."
    )
    return "\n".join(out)


def _render_top_json(rows: tuple[ExpensiveResponse, ...], by: str) -> str:
    payload = {
        "sort_by": by,
        "rows": [
            {
                "id": r.id,
                "datetime_utc": r.datetime_utc,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "source": r.source,
                "prompt_preview": r.prompt_preview,
            }
            for r in rows
        ],
    }
    return json.dumps(payload, indent=2)


def _render_daily_json(days: tuple[DailyRow, ...], head: Headlines) -> str:
    payload = {
        "days": [
            {
                "day": d.day.isoformat(),
                "responses": d.responses,
                "input_tokens": d.input_tokens,
                "output_tokens": d.output_tokens,
                "cost_usd": d.cost_usd,
            }
            for d in days
        ],
        "headlines": {
            "today": head.today,
            "this_week": head.this_week,
            "this_month": head.this_month,
            "all_time": head.all_time,
            "top_models_month": [
                {"model": r.model, "cost_usd": r.best_cost_usd}
                for r in head.top_models_month
            ],
        },
    }
    return json.dumps(payload, indent=2)


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
                "variants": [
                    {"model": raw, "resolved_model": resolved}
                    for raw, resolved in r.variants
                ],
                "responses": r.response_count,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "logged_cost_usd": r.logged_cost_usd,
                "priced_cost_usd": r.priced_cost_usd,
                "best_cost_usd": r.best_cost_usd,
                "priced": r.priced,
                "source": r.source,
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
        """Report token usage and spend from the llm logs database.

        With no arguments, renders a 14-day sparkline + today / this
        week / this month / all-time headlines + top-3 models this
        month. Pass --since / --until / --days for a per-model table
        over a specific window; use `llm cost today`, `llm cost
        yesterday`, or `llm cost all` for common shorthands.
        """
        if ctx.invoked_subcommand is not None:
            return

        # Bare `llm cost`: the cute default landing.
        if days is None and since is None and until is None and model_glob is None:
            prices = _load_price_table(prices_path)
            amap = _alias_map()
            db = sqlite_utils.Database(str(db_path or _logs_db_path()))
            daily = daily_summary(db, days=14, prices=prices, alias_map=amap)
            head = headlines(db, prices=prices, alias_map=amap)
            if as_json:
                click.echo(_render_daily_json(daily, head))
            else:
                click.echo(_render_daily(daily, head))
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
            # Only --model passed; show all-time for that model.
            start = end = None
            label = "all time"

        _report(start, end, label, model_glob, prices_path, as_json, db_path)

    @cost_group.command(name="all")
    @_shared_options
    def cost_all(
        model_glob: str | None,
        prices_path: Path | None,
        as_json: bool,
        db_path: Path | None,
    ) -> None:
        """Per-model spend across all logged responses (escape hatch)."""
        _report(None, None, "all time", model_glob, prices_path, as_json, db_path)

    @cost_group.command(name="top")
    @click.option("--limit", "-n", type=int, default=10, help="How many rows (default 10).")
    @click.option(
        "--by",
        type=click.Choice(["cost", "input", "output", "total"]),
        default="cost",
        help="Sort key: cost, input tokens, output tokens, or total tokens.",
    )
    @click.option("--since", type=str, help="Start date YYYY-MM-DD.")
    @click.option("--until", type=str, help="End date YYYY-MM-DD.")
    @click.option("--days", type=int, help="Last N days (including today).")
    @_shared_options
    def cost_top(
        limit: int,
        by: str,
        since: str | None,
        until: str | None,
        days: int | None,
        model_glob: str | None,
        prices_path: Path | None,
        as_json: bool,
        db_path: Path | None,
    ) -> None:
        """Show the most expensive individual responses.

        Useful for catching accidents — prompts where a whole file or
        directory got piped in without realising. Pair with
        `llm-confirm-tokens` to head them off before they send.
        """
        if days is not None:
            today = _today_local()
            start, _ = local_day_bounds(today - timedelta(days=days - 1))
            _, end = local_day_bounds(today)
        else:
            start = local_day_bounds(_parse_date(since))[0] if since else None
            end = local_day_bounds(_parse_date(until))[1] if until else None

        prices = _load_price_table(prices_path)
        amap = _alias_map()
        db = sqlite_utils.Database(str(db_path or _logs_db_path()))
        rows = top_responses(
            db,
            limit=limit,
            by=by,
            since=start,
            until=end,
            model_glob=model_glob,
            prices=prices,
            alias_map=amap,
        )

        if as_json:
            click.echo(_render_top_json(rows, by))
        else:
            click.echo(_render_top(rows, by))

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
