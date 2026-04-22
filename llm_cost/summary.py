"""Aggregate token usage and cost from the llm logs database."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo

import sqlite_utils

from .pricing import Price, _canonical, default_prices, resolve


@dataclass(frozen=True)
class ModelUsage:
    model: str  # canonical display name (post alias resolution)
    variants: tuple[tuple[str, str | None], ...]  # raw (model, resolved_model) pairs that rolled up
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


def local_day_bounds(day: date, tz: tzinfo | None = None) -> tuple[datetime, datetime]:
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


def canonical_key(
    model: str,
    resolved_model: str | None = None,
    alias_map: dict[str, str] | None = None,
) -> str:
    """Derive a stable grouping key for a logged (model, resolved_model) pair.

    Preference order:
      1. Alias-map hit on ``resolved_model`` — returns the Model's
         ``.model_id`` so e.g. ``claude-haiku-4.5`` collapses to
         ``anthropic/claude-haiku-4-5-20251001``.
      2. Alias-map hit on ``model`` — same.
      3. ``_canonical(resolved_model)`` when resolved is set — strips
         provider prefix + date/variant suffixes for retired models
         that are no longer in the alias map.
      4. ``_canonical(model)``.

    Keys returned from (1)/(2) are full provider-prefixed model ids;
    keys from (3)/(4) are the stripped shorter form. Both flow into
    ``pricing.resolve`` which applies ``_canonical`` internally, so
    either shape hits the same price-table row.
    """
    amap = alias_map or {}
    for candidate in (resolved_model, model):
        if candidate and candidate in amap:
            return amap[candidate]
    if resolved_model:
        return _canonical(resolved_model)
    return _canonical(model)


def summarise(
    db: sqlite_utils.Database,
    since: datetime | None = None,
    until: datetime | None = None,
    model_glob: str | None = None,
    prices: dict[str, Price] | None = None,
    alias_map: dict[str, str] | None = None,
) -> Summary:
    """Aggregate ``responses`` into per-model token/cost rows.

    - ``since`` / ``until`` are UTC half-open bounds (``since <= t < until``).
      Pass naive ``datetime``s at your peril — use ``local_day_bounds``
      or construct timezone-aware values.
    - ``model_glob`` is a SQL LIKE pattern applied to the raw ``model`` column.
    - ``prices`` overrides the bundled price table.
    - ``alias_map`` maps alias/model-name to canonical ``model_id`` —
      typically ``{name: m.model_id for name, m in llm.get_model_aliases().items()}``.
      When None or empty, folding falls back to the prefix/suffix
      stripping heuristic only.
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
    """

    # Fold SQL rows into canonical groups in Python — the alias map lives
    # in llm's runtime registry, not the DB, so we can't express this in SQL.
    groups: dict[str, dict] = {}
    for row in db.execute(sql, params):
        model, resolved, count, inp, outp, logged_cost = row
        resolved_opt = resolved or None
        key = canonical_key(model, resolved_opt, alias_map)
        g = groups.setdefault(
            key,
            {
                "variants": set(),
                "count": 0,
                "input": 0,
                "output": 0,
                "logged": 0.0,
            },
        )
        g["variants"].add((model, resolved_opt))
        g["count"] += int(count)
        g["input"] += int(inp)
        g["output"] += int(outp)
        g["logged"] += float(logged_cost)

    rows: list[ModelUsage] = []
    for key, g in groups.items():
        price = resolve(key, None, table)
        if price is not None:
            priced_cost = price.cost(g["input"], g["output"])
            priced = True
        else:
            priced_cost = 0.0
            priced = False
        rows.append(
            ModelUsage(
                model=key,
                variants=tuple(sorted(g["variants"], key=lambda v: (v[0], v[1] or ""))),
                response_count=g["count"],
                input_tokens=g["input"],
                output_tokens=g["output"],
                logged_cost_usd=g["logged"],
                priced_cost_usd=priced_cost,
                priced=priced,
            )
        )

    # Highest token volume first, matching the old SQL ORDER BY.
    rows.sort(key=lambda r: r.input_tokens + r.output_tokens, reverse=True)

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
    """Yield canonical names that had no price match and no logged cost."""
    seen: set[str] = set()
    for row in summary.rows:
        if row.priced or row.logged_cost_usd > 0:
            continue
        if row.model in seen:
            continue
        seen.add(row.model)
        yield row.model
