"""Aggregate token usage and cost from the llm logs database."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo

import sqlite_utils

from .pricing import Price, _canonical, default_prices, resolve


def _today_local() -> date:
    return datetime.now().astimezone().date()


@dataclass(frozen=True)
class ModelUsage:
    model: str  # canonical display name (post alias resolution)
    variants: tuple[tuple[str, str | None], ...]  # raw (model, resolved_model) pairs that rolled up
    response_count: int
    input_tokens: int
    output_tokens: int
    logged_cost_usd: float  # sum of responses.cost_usd when present
    priced_cost_usd: float  # priced over the group's aggregate tokens
    best_cost_usd: float  # sum of per-subgroup (logged > 0 ? logged : priced)
    priced: bool  # False when no price was found for this model
    source: str  # "logged" | "priced" | "mixed" | "unpriced"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


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
        return any(r.source == "unpriced" for r in self.rows)


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
    # Per-subgroup "best cost" (logged if llm wrote one, else priced) is
    # summed into the group so mixed groups — some subgroups logged, some
    # unlogged — account for both halves. Without this the priced tokens
    # from unlogged subgroups would silently drop out whenever any other
    # subgroup had a logged value.
    groups: dict[str, dict] = {}
    for row in db.execute(sql, params):
        model, resolved, count, inp, outp, logged_cost = row
        resolved_opt = resolved or None
        key = canonical_key(model, resolved_opt, alias_map)
        price = resolve(key, None, table)
        subgroup_priced = price.cost(int(inp), int(outp)) if price else 0.0
        subgroup_logged = float(logged_cost)
        subgroup_best = subgroup_logged if subgroup_logged > 0 else subgroup_priced
        g = groups.setdefault(
            key,
            {
                "variants": set(),
                "count": 0,
                "input": 0,
                "output": 0,
                "logged": 0.0,
                "best": 0.0,
                "logged_subgroups": 0,
                "priced_subgroups": 0,
                "unpriced_subgroups": 0,
            },
        )
        g["variants"].add((model, resolved_opt))
        g["count"] += int(count)
        g["input"] += int(inp)
        g["output"] += int(outp)
        g["logged"] += subgroup_logged
        g["best"] += subgroup_best
        if subgroup_logged > 0:
            g["logged_subgroups"] += 1
        elif price is not None:
            g["priced_subgroups"] += 1
        else:
            g["unpriced_subgroups"] += 1

    rows: list[ModelUsage] = []
    for key, g in groups.items():
        price = resolve(key, None, table)
        priced_cost = price.cost(g["input"], g["output"]) if price else 0.0
        has_price = price is not None

        if g["logged_subgroups"] and (g["priced_subgroups"] or g["unpriced_subgroups"]):
            source = "mixed"
        elif g["logged_subgroups"]:
            source = "logged"
        elif g["priced_subgroups"]:
            source = "priced"
        else:
            source = "unpriced"

        rows.append(
            ModelUsage(
                model=key,
                variants=tuple(sorted(g["variants"], key=lambda v: (v[0], v[1] or ""))),
                response_count=g["count"],
                input_tokens=g["input"],
                output_tokens=g["output"],
                logged_cost_usd=g["logged"],
                priced_cost_usd=priced_cost,
                best_cost_usd=g["best"],
                priced=has_price,
                source=source,
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


@dataclass(frozen=True)
class ExpensiveResponse:
    id: str
    datetime_utc: str  # raw llm string, e.g. "2026-04-21T14:22:00.123456+00:00"
    model: str  # canonical display name
    input_tokens: int
    output_tokens: int
    cost_usd: float  # logged if >0, else priced
    source: str  # "logged" | "priced" | "unpriced"
    prompt_preview: str  # single-line snippet, caller-controlled width

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _compact_preview(text: str, width: int) -> str:
    """Collapse whitespace to single spaces, trim to ``width``, ellipsis if cut."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= width:
        return cleaned
    return cleaned[: max(1, width - 1)] + "…"


def top_responses(
    db: sqlite_utils.Database,
    limit: int = 10,
    by: str = "cost",
    since: datetime | None = None,
    until: datetime | None = None,
    model_glob: str | None = None,
    prices: dict[str, Price] | None = None,
    alias_map: dict[str, str] | None = None,
    preview_chars: int = 80,
) -> tuple[ExpensiveResponse, ...]:
    """Return the ``limit`` most expensive individual responses.

    ``by`` picks the sort key: ``cost`` (best_cost_usd), ``input``
    (input_tokens), ``output`` (output_tokens), or ``total``
    (input + output).

    The SQL pre-filters a larger candidate set ordered by a cheap
    combined heuristic (logged cost + token totals); the precise cost
    is then computed in Python per candidate using the same price-table
    / alias-map logic as ``summarise``.
    """
    if by not in {"cost", "input", "output", "total"}:
        raise ValueError(f"by must be cost/input/output/total, got {by!r}")
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
    candidate_limit = max(limit * 20, 100)

    # Prefilter in SQL by a weighted combo so we don't miss either
    # high-cost (logged) or high-token (unlogged) candidates.
    sql = f"""
        SELECT
            id, datetime_utc, model, COALESCE(resolved_model, '') AS resolved_model,
            COALESCE(input_tokens, 0) AS input_tokens,
            COALESCE(output_tokens, 0) AS output_tokens,
            COALESCE(cost_usd, 0) AS logged_cost,
            COALESCE(prompt, '') AS prompt
        FROM responses
        {where}
        ORDER BY (
            COALESCE(cost_usd, 0) * 1000
            + COALESCE(input_tokens, 0)
            + COALESCE(output_tokens, 0)
        ) DESC
        LIMIT ?
    """
    params.append(candidate_limit)

    rows: list[ExpensiveResponse] = []
    for row in db.execute(sql, params):
        rid, dt, model, resolved, inp, outp, logged, prompt = row
        key = canonical_key(model, resolved or None, alias_map)
        price = resolve(key, None, table)
        logged_f = float(logged)
        if logged_f > 0:
            cost = logged_f
            source = "logged"
        elif price is not None:
            cost = price.cost(int(inp), int(outp))
            source = "priced"
        else:
            cost = 0.0
            source = "unpriced"
        rows.append(
            ExpensiveResponse(
                id=rid,
                datetime_utc=dt,
                model=key,
                input_tokens=int(inp),
                output_tokens=int(outp),
                cost_usd=cost,
                source=source,
                prompt_preview=_compact_preview(prompt or "", preview_chars),
            )
        )

    sort_keys = {
        "cost": lambda r: r.cost_usd,
        "input": lambda r: r.input_tokens,
        "output": lambda r: r.output_tokens,
        "total": lambda r: r.total_tokens,
    }
    rows.sort(key=sort_keys[by], reverse=True)
    return tuple(rows[:limit])


@dataclass(frozen=True)
class DailyRow:
    day: date
    responses: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class Headlines:
    """Headline spend numbers for the default ``llm cost`` landing."""

    today: float
    this_week: float  # trailing 7 days inclusive of today
    this_month: float  # calendar month-to-date
    all_time: float
    top_models_month: tuple[ModelUsage, ...]


def daily_summary(
    db: sqlite_utils.Database,
    days: int = 14,
    prices: dict[str, Price] | None = None,
    alias_map: dict[str, str] | None = None,
    today: date | None = None,
) -> tuple[DailyRow, ...]:
    """Per-day spend for the trailing ``days`` days (inclusive of today).

    Empty days are emitted with zeros so the sparkline keeps a stable
    width. ``today`` is injectable for tests.
    """
    today = today or _today_local()
    start_day = today - timedelta(days=days - 1)
    start_utc, _ = local_day_bounds(start_day)
    _, end_utc = local_day_bounds(today)
    table = prices if prices is not None else default_prices()

    sql = """
        SELECT
            date(datetime_utc, 'localtime') AS local_date,
            model,
            COALESCE(resolved_model, '') AS resolved_model,
            COUNT(*) AS n,
            COALESCE(SUM(input_tokens), 0) AS input_tokens,
            COALESCE(SUM(output_tokens), 0) AS output_tokens,
            COALESCE(SUM(cost_usd), 0) AS logged_cost
        FROM responses
        WHERE datetime_utc >= ? AND datetime_utc < ?
        GROUP BY local_date, model, resolved_model
    """

    buckets: dict[str, dict] = {}
    for row in db.execute(sql, [_iso(start_utc), _iso(end_utc)]):
        local_date, model, resolved, n, inp, outp, logged = row
        key = canonical_key(model, resolved or None, alias_map)
        price = resolve(key, None, table)
        priced_cost = price.cost(int(inp), int(outp)) if price else 0.0
        best = float(logged) if float(logged) > 0 else priced_cost
        b = buckets.setdefault(
            local_date,
            {"responses": 0, "input": 0, "output": 0, "cost": 0.0},
        )
        b["responses"] += int(n)
        b["input"] += int(inp)
        b["output"] += int(outp)
        b["cost"] += best

    rows: list[DailyRow] = []
    d = start_day
    while d <= today:
        b = buckets.get(d.isoformat(), {"responses": 0, "input": 0, "output": 0, "cost": 0.0})
        rows.append(
            DailyRow(
                day=d,
                responses=b["responses"],
                input_tokens=b["input"],
                output_tokens=b["output"],
                cost_usd=b["cost"],
            )
        )
        d += timedelta(days=1)
    return tuple(rows)


def headlines(
    db: sqlite_utils.Database,
    prices: dict[str, Price] | None = None,
    alias_map: dict[str, str] | None = None,
    today: date | None = None,
    top_n: int = 3,
) -> Headlines:
    """Headline totals (today / this week / this month / all time) plus
    top-N models for the current calendar month."""
    today = today or _today_local()
    _, today_end = local_day_bounds(today)

    today_start, _ = local_day_bounds(today)
    week_start, _ = local_day_bounds(today - timedelta(days=6))
    month_start, _ = local_day_bounds(today.replace(day=1))

    def _cost(since):
        return summarise(
            db, since=since, until=today_end, prices=prices, alias_map=alias_map
        ).total_cost_usd

    month_summary = summarise(
        db, since=month_start, until=today_end, prices=prices, alias_map=alias_map
    )
    all_time = summarise(db, prices=prices, alias_map=alias_map).total_cost_usd

    top = tuple(
        sorted(month_summary.rows, key=lambda r: r.best_cost_usd, reverse=True)[:top_n]
    )

    return Headlines(
        today=_cost(today_start),
        this_week=_cost(week_start),
        this_month=month_summary.total_cost_usd,
        all_time=all_time,
        top_models_month=top,
    )


@dataclass(frozen=True)
class DupeRow:
    model: str  # canonical
    dupe_groups: int  # distinct request_keys where ≥2 real API calls fired
    extra_calls: int  # total_real_calls in dupe_groups − dupe_groups
    wasted_usd: float  # sum over groups of (group_total_cost − first_call_cost)


@dataclass(frozen=True)
class DupeReport:
    rows: tuple[DupeRow, ...]
    total_groups: int
    total_extra_calls: int
    total_wasted_usd: float
    indexed_responses: int  # responses from the window that have a replay_index row
    total_responses: int  # total responses in the window
    replay_index_present: bool


def dupe_report(
    db: sqlite_utils.Database,
    since: datetime | None = None,
    until: datetime | None = None,
    model_glob: str | None = None,
    prices: dict[str, Price] | None = None,
    alias_map: dict[str, str] | None = None,
) -> DupeReport:
    """Identify duplicate requests that could have been replayed.

    Joins ``responses`` against ``replay_index`` (written by llm-replay)
    on ``request_key``. A dupe group is a set of ≥2 real API calls
    (``replay_of IS NULL``) sharing the same request_key. Savings per
    group assume you kept the first call and replayed the rest:
    ``sum(cost) − first_call_cost``.

    If ``replay_index`` doesn't exist (llm-replay not installed), the
    report returns empty with ``replay_index_present=False`` so the
    caller can advise the user.
    """
    table = prices if prices is not None else default_prices()

    total_count = _count_responses(db, since, until, model_glob)

    if not db["replay_index"].exists():
        return DupeReport(
            rows=(),
            total_groups=0,
            total_extra_calls=0,
            total_wasted_usd=0.0,
            indexed_responses=0,
            total_responses=total_count,
            replay_index_present=False,
        )

    clauses = ["r.replay_of IS NULL"]
    params: list[object] = []
    if since is not None:
        clauses.append("r.datetime_utc >= ?")
        params.append(_iso(since))
    if until is not None:
        clauses.append("r.datetime_utc < ?")
        params.append(_iso(until))
    if model_glob:
        clauses.append("r.model LIKE ?")
        params.append(model_glob)

    where = " AND ".join(clauses)

    indexed_count_sql = f"""
        SELECT COUNT(*) FROM responses r
        JOIN replay_index i ON i.response_id = r.id
        WHERE {where}
    """
    indexed_count = next(iter(db.execute(indexed_count_sql, params)))[0]

    # Pull every real-API-call row that shares a request_key with ≥1 other
    # real-API-call row in the same window. HAVING inside the subquery is
    # what limits us to actual dupes instead of every indexed row.
    sql = f"""
        SELECT
            i.request_key,
            r.id,
            r.model,
            COALESCE(r.resolved_model, '') AS resolved_model,
            r.datetime_utc,
            COALESCE(r.input_tokens, 0) AS input_tokens,
            COALESCE(r.output_tokens, 0) AS output_tokens,
            COALESCE(r.cost_usd, 0) AS logged_cost
        FROM replay_index i
        JOIN responses r ON r.id = i.response_id
        WHERE {where}
          AND i.request_key IN (
              SELECT i2.request_key
              FROM replay_index i2
              JOIN responses r2 ON r2.id = i2.response_id
              WHERE {where.replace('r.', 'r2.')}
              GROUP BY i2.request_key
              HAVING COUNT(*) > 1
          )
        ORDER BY i.request_key, r.datetime_utc
    """
    # The subquery reuses the same filters; duplicate the params list.
    groups_by_key: dict[str, list[tuple]] = {}
    for row in db.execute(sql, params + params):
        key = row[0]
        groups_by_key.setdefault(key, []).append(row)

    per_model: dict[str, dict] = {}
    total_wasted = 0.0
    total_extra = 0
    for group in groups_by_key.values():
        # Pick the first call (already date-sorted) — that's the "kept" one.
        first = group[0]
        rest = group[1:]
        group_model = canonical_key(first[2], first[3] or None, alias_map)
        group_price = resolve(group_model, None, table)
        wasted = sum(_row_cost(r, group_price) for r in rest)
        bucket = per_model.setdefault(
            group_model,
            {"groups": 0, "extra_calls": 0, "wasted": 0.0},
        )
        bucket["groups"] += 1
        bucket["extra_calls"] += len(rest)
        bucket["wasted"] += wasted
        total_wasted += wasted
        total_extra += len(rest)

    rows = tuple(
        sorted(
            (
                DupeRow(
                    model=m,
                    dupe_groups=b["groups"],
                    extra_calls=b["extra_calls"],
                    wasted_usd=b["wasted"],
                )
                for m, b in per_model.items()
            ),
            key=lambda r: r.wasted_usd,
            reverse=True,
        )
    )

    return DupeReport(
        rows=rows,
        total_groups=len(groups_by_key),
        total_extra_calls=total_extra,
        total_wasted_usd=total_wasted,
        indexed_responses=int(indexed_count),
        total_responses=total_count,
        replay_index_present=True,
    )


def _row_cost(row: tuple, price: Price | None) -> float:
    """Cost for a single dupe-report row: logged if > 0 else priced."""
    _, _, _, _, _, inp, outp, logged = row
    logged_f = float(logged)
    if logged_f > 0:
        return logged_f
    return price.cost(int(inp), int(outp)) if price else 0.0


def _count_responses(
    db: sqlite_utils.Database,
    since: datetime | None,
    until: datetime | None,
    model_glob: str | None,
) -> int:
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
    sql = f"SELECT COUNT(*) FROM responses {where}"
    return int(next(iter(db.execute(sql, params)))[0])


def models_without_prices(summary: Summary) -> Iterable[str]:
    """Yield canonical names for groups with no price hit *and* no logged cost."""
    seen: set[str] = set()
    for row in summary.rows:
        if row.source != "unpriced":
            continue
        if row.model in seen:
            continue
        seen.add(row.model)
        yield row.model
