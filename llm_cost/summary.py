"""Aggregate token usage and cost from the llm logs database."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

import sqlite_utils

from .pricing import Price, default_prices, resolve


@dataclass(frozen=True)
class ModelUsage:
    model: str
    resolved_model: str | None
    response_count: int
    input_tokens: int
    output_tokens: int
    logged_cost_usd: float  # sum of responses.cost_usd when present
    priced_cost_usd: float  # computed from the price table
    priced: bool  # False when no price was found for this model

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def best_cost_usd(self) -> float:
        """Prefer the value llm itself logged; otherwise fall back to priced."""
        return self.logged_cost_usd if self.logged_cost_usd > 0 else self.priced_cost_usd


@dataclass(frozen=True)
class Summary:
    since_utc: datetime | None
    until_utc: datetime | None
    rows: tuple[ModelUsage, ...]

    @property
    def total_input(self) -> int:
        return sum(r.input_tokens for r in self.rows)

    @property
    def total_output(self) -> int:
        return sum(r.output_tokens for r in self.rows)

    @property
    def total_responses(self) -> int:
        return sum(r.response_count for r in self.rows)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.best_cost_usd for r in self.rows)

    @property
    def has_unpriced(self) -> bool:
        return any(not r.priced and r.logged_cost_usd == 0 for r in self.rows)


def local_day_bounds(day: date, tz: timezone | None = None) -> tuple[datetime, datetime]:
    """Return the UTC half-open interval [start, end) that covers ``day``
    in the given timezone (local by default).

    llm stores ``datetime_utc`` as UTC ISO strings, so we compare against
    these UTC bounds to get a human "today" that matches the user's wall
    clock rather than the date in London.
    """
    if tz is None:
        tz = datetime.now(timezone.utc).astimezone().tzinfo  # local tz
    start_local = datetime.combine(day, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def summarise(
    db: sqlite_utils.Database,
    since: datetime | None = None,
    until: datetime | None = None,
    model_glob: str | None = None,
    prices: dict[str, Price] | None = None,
) -> Summary:
    """Aggregate ``responses`` into per-model token/cost rows.

    - ``since`` / ``until`` are UTC half-open bounds (``since <= t < until``).
      Pass naive ``datetime``s at your peril — use ``local_day_bounds``
      or construct timezone-aware values.
    - ``model_glob`` is a SQL LIKE pattern applied to ``model``.
    - ``prices`` overrides ``pricing.DEFAULT_PRICES``.
    """
    table = prices if prices is not None else default_prices()

    clauses: list[str] = []
    params: list[object] = []
    if since is not None:
        clauses.append("datetime_utc >= ?")
        params.append(_iso(since))
    if until is not None:
        clauses.append("datetime_utc < ?")
        params.append(_iso(until))
    if model_glob:
        clauses.append("model LIKE ?")
        params.append(model_glob)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = f"""
        SELECT
            model,
            COALESCE(resolved_model, '') AS resolved_model,
            COUNT(*) AS response_count,
            COALESCE(SUM(input_tokens), 0) AS input_tokens,
            COALESCE(SUM(output_tokens), 0) AS output_tokens,
            COALESCE(SUM(cost_usd), 0) AS logged_cost_usd
        FROM responses
        {where}
        GROUP BY model, resolved_model
        ORDER BY (COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(output_tokens), 0)) DESC
    """

    rows: list[ModelUsage] = []
    for row in db.execute(sql, params):
        model, resolved, count, inp, outp, logged_cost = row
        price = resolve(model, resolved or None, table)
        if price is not None:
            priced_cost = price.cost(int(inp), int(outp))
            priced = True
        else:
            priced_cost = 0.0
            priced = False
        rows.append(
            ModelUsage(
                model=model,
                resolved_model=resolved or None,
                response_count=int(count),
                input_tokens=int(inp),
                output_tokens=int(outp),
                logged_cost_usd=float(logged_cost),
                priced_cost_usd=priced_cost,
                priced=priced,
            )
        )

    return Summary(since_utc=since, until_utc=until, rows=tuple(rows))


def _iso(dt: datetime) -> str:
    """Format a UTC datetime to match llm's ``datetime_utc`` column.

    llm writes ``datetime.utcnow().isoformat()`` plus a ``+00:00`` suffix;
    we normalise to the same shape so string comparisons hit the index.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def models_without_prices(summary: Summary) -> Iterable[str]:
    """Yield distinct model names that had no price match and no logged cost."""
    seen: set[str] = set()
    for row in summary.rows:
        if row.priced or row.logged_cost_usd > 0:
            continue
        key = row.resolved_model or row.model
        if key in seen:
            continue
        seen.add(key)
        yield key
