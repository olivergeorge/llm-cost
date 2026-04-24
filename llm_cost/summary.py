"""Aggregate token usage and cost from the llm logs database."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo

import sqlite_utils

from .pricing import Price, TokenBreakdown, _canonical, default_prices, resolve


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


def parse_token_details(
    raw: str | None,
    input_tokens: int,
    output_tokens: int,
) -> TokenBreakdown:
    """Split ``input_tokens`` / ``output_tokens`` into per-bucket counts.

    Recognises the Gemini ``token_details`` payload that ``llm-gemini``
    writes into ``responses.token_details``. Example::

        {"candidatesTokenCount": 1189,
         "cachedContentTokenCount": 12265,
         "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 16342}],
         "cacheTokensDetails":  [{"modality": "TEXT", "tokenCount": 12265}],
         "thoughtsTokenCount": 790}

    Bucketing rules (empirically verified against live logs):

    - ``input_tokens`` column ≡ ``sum(promptTokensDetails.tokenCount)``
      + ``toolUsePromptTokenCount``. Cached tokens are a **subset** of
      ``promptTokensDetails``, not additive.
    - ``output_tokens`` column ≡ ``candidatesTokenCount`` +
      ``thoughtsTokenCount``.

    The parser therefore subtracts cached from the matching modality
    (via ``cacheTokensDetails``) so we don't double-count, and treats
    tool-use prompt tokens as text. When ``raw`` is ``None``, empty,
    malformed, or of an unknown shape, falls back to
    :meth:`TokenBreakdown.text_only` using the aggregate columns.
    """
    if not raw:
        return TokenBreakdown.text_only(input_tokens, output_tokens)
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return TokenBreakdown.text_only(input_tokens, output_tokens)
    if not isinstance(data, dict):
        return TokenBreakdown.text_only(input_tokens, output_tokens)

    # Only the Gemini shape is modelled right now. Other providers (or
    # future fields we don't recognise) fall through to text-only.
    has_gemini_fields = any(
        k in data
        for k in ("promptTokensDetails", "candidatesTokenCount", "thoughtsTokenCount")
    )
    if not has_gemini_fields:
        return TokenBreakdown.text_only(input_tokens, output_tokens)

    text = 0
    audio = 0
    for entry in data.get("promptTokensDetails") or ():
        if not isinstance(entry, dict):
            continue
        tokens = int(entry.get("tokenCount") or 0)
        modality = str(entry.get("modality") or "TEXT").upper()
        if modality == "AUDIO":
            audio += tokens
        else:
            # TEXT, IMAGE, VIDEO — price as text. LiteLLM doesn't yet
            # publish per-modality rates for images/video on Gemini.
            text += tokens

    # Tool-use prompt tokens aren't broken down by modality. Treat as text.
    text += int(data.get("toolUsePromptTokenCount") or 0)

    cached_total = int(data.get("cachedContentTokenCount") or 0)
    cached_audio = 0
    for entry in data.get("cacheTokensDetails") or ():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("modality") or "TEXT").upper() == "AUDIO":
            cached_audio += int(entry.get("tokenCount") or 0)
    cached_text = cached_total - cached_audio
    # Cached tokens are a subset of promptTokensDetails — subtract so we
    # don't double-count them under both a modality bucket and the
    # cache bucket.
    text = max(0, text - cached_text)
    audio = max(0, audio - cached_audio)

    # Reconcile against the authoritative aggregate column: if our
    # computed total differs (unknown field, malformed entry), absorb
    # the residual into text so the bucket sum still matches
    # ``input_tokens`` exactly.
    residual = input_tokens - (text + audio + cached_total)
    if residual:
        text = max(0, text + residual)

    reasoning = int(data.get("thoughtsTokenCount") or 0)
    base_output = max(0, output_tokens - reasoning)

    return TokenBreakdown(
        input_text=text,
        input_audio=audio,
        input_cached=cached_total,
        output=base_output,
        output_reasoning=reasoning,
    )


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

    # Per-row fetch: we need ``token_details`` to split input/output into
    # text/audio/cached/reasoning buckets. SQL-side aggregation can't
    # express that split cleanly (the audio count is nested inside a
    # JSON array), so we aggregate in Python. For typical logs.db sizes
    # this is still milliseconds.
    sql = f"""
        SELECT
            model,
            COALESCE(resolved_model, '') AS resolved_model,
            COALESCE(input_tokens, 0) AS input_tokens,
            COALESCE(output_tokens, 0) AS output_tokens,
            COALESCE(cost_usd, 0) AS logged_cost_usd,
            token_details
        FROM responses
        {where}
    """

    groups: dict[str, dict] = {}
    for row in db.execute(sql, params):
        model, resolved, inp, outp, logged_cost, token_details = row
        resolved_opt = resolved or None
        key = canonical_key(model, resolved_opt, alias_map)
        price = resolve(key, None, table)
        breakdown = parse_token_details(token_details, int(inp), int(outp))
        row_priced = price.cost_for(breakdown) if price else 0.0
        row_logged = float(logged_cost)
        row_best = row_logged if row_logged > 0 else row_priced
        g = groups.setdefault(
            key,
            {
                "variants": set(),
                "count": 0,
                "input": 0,
                "output": 0,
                "breakdown": TokenBreakdown(0, 0, 0, 0, 0),
                "logged": 0.0,
                "best": 0.0,
                "logged_rows": 0,
                "priced_rows": 0,
                "unpriced_rows": 0,
            },
        )
        g["variants"].add((model, resolved_opt))
        g["count"] += 1
        g["input"] += int(inp)
        g["output"] += int(outp)
        g["breakdown"] = g["breakdown"] + breakdown
        g["logged"] += row_logged
        g["best"] += row_best
        if row_logged > 0:
            g["logged_rows"] += 1
        elif price is not None:
            g["priced_rows"] += 1
        else:
            g["unpriced_rows"] += 1

    rows: list[ModelUsage] = []
    for key, g in groups.items():
        price = resolve(key, None, table)
        priced_cost = price.cost_for(g["breakdown"]) if price else 0.0
        has_price = price is not None

        if g["logged_rows"] and (g["priced_rows"] or g["unpriced_rows"]):
            source = "mixed"
        elif g["logged_rows"]:
            source = "logged"
        elif g["priced_rows"]:
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
            COALESCE(prompt, '') AS prompt,
            token_details
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
        rid, dt, model, resolved, inp, outp, logged, prompt, token_details = row
        key = canonical_key(model, resolved or None, alias_map)
        price = resolve(key, None, table)
        logged_f = float(logged)
        if logged_f > 0:
            cost = logged_f
            source = "logged"
        elif price is not None:
            breakdown = parse_token_details(token_details, int(inp), int(outp))
            cost = price.cost_for(breakdown)
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
    this_week: float  # calendar week-to-date (Monday start, inclusive of today)
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

    # Per-row fetch so we can split audio/cached via token_details.
    # `date(datetime_utc, 'localtime')` is computed in SQL so timezone
    # handling matches the prior grouping semantics.
    sql = """
        SELECT
            date(datetime_utc, 'localtime') AS local_date,
            model,
            COALESCE(resolved_model, '') AS resolved_model,
            COALESCE(input_tokens, 0) AS input_tokens,
            COALESCE(output_tokens, 0) AS output_tokens,
            COALESCE(cost_usd, 0) AS logged_cost,
            token_details
        FROM responses
        WHERE datetime_utc >= ? AND datetime_utc < ?
    """

    buckets: dict[str, dict] = {}
    for row in db.execute(sql, [_iso(start_utc), _iso(end_utc)]):
        local_date, model, resolved, inp, outp, logged, token_details = row
        key = canonical_key(model, resolved or None, alias_map)
        price = resolve(key, None, table)
        if price:
            breakdown = parse_token_details(token_details, int(inp), int(outp))
            priced_cost = price.cost_for(breakdown)
        else:
            priced_cost = 0.0
        best = float(logged) if float(logged) > 0 else priced_cost
        b = buckets.setdefault(
            local_date,
            {"responses": 0, "input": 0, "output": 0, "cost": 0.0},
        )
        b["responses"] += 1
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
    # Monday-start ISO week so the week and month rows agree on calendar
    # semantics: "This week" now parallels "This month" as calendar-to-date,
    # not a trailing 7-day window.
    week_start, _ = local_day_bounds(today - timedelta(days=today.weekday()))
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
    dupe_groups: int  # distinct fingerprints where ≥2 calls fired
    extra_calls: int  # total calls in dupe groups − dupe_groups
    wasted_usd: float  # sum over groups of (group_total_cost − first_call_cost)


@dataclass(frozen=True)
class DupeReport:
    rows: tuple[DupeRow, ...]
    total_groups: int
    total_extra_calls: int
    total_wasted_usd: float
    total_responses: int  # responses considered in the window


def dupe_report(
    db: sqlite_utils.Database,
    since: datetime | None = None,
    until: datetime | None = None,
    model_glob: str | None = None,
    prices: dict[str, Price] | None = None,
    alias_map: dict[str, str] | None = None,
) -> DupeReport:
    """Identify duplicate requests by fingerprinting core request inputs.

    Two calls are dupes if everything the LLM sees is identical: model,
    system prompt, user prompt, options, schema, prior conversation
    turns, attachments, and fragments. llm content-addresses attachments
    (``attachments.id`` = SHA-256 of content) and fragments
    (``fragments.hash``), so "identical file uploaded twice" folds to
    the same key for free.

    The first call in a group is the one you'd keep; savings assume the
    rest were replayed: ``sum(cost) − first_call_cost`` per group.

    Uses only the core llm schema — no plugin dependency.
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

    total_count = _count_responses(db, since, until, model_glob)

    # Fingerprint strategy: fetch the window rows plus the ancillary
    # pieces (prior conversation turns, attachments, fragments) in
    # batch queries, then assemble fingerprints in Python. Doing it in
    # SQL via correlated subqueries is O(n²) against the full
    # ``responses`` table on a logs.db with no conversation_id index,
    # which takes minutes on realistic histories.
    #
    # Separator bytes: x'1d' between fields, x'1f' between list items,
    # x'1e' between the (prompt, response) halves of a prior turn.
    # Rare control bytes keep collisions theoretically possible but
    # not practical.
    #
    # llm content-addresses attachments (``attachments.id`` = sha256)
    # and fragments (``fragments.hash``), so identical-content uploads
    # fold to the same key automatically.
    row_sql = f"""
        SELECT
            id, datetime_utc, conversation_id,
            model, COALESCE(resolved_model, '') AS resolved_model,
            COALESCE(system, '') AS system,
            COALESCE(prompt, '') AS prompt,
            COALESCE(options_json, '{{}}') AS options_json,
            COALESCE(schema_id, '') AS schema_id,
            COALESCE(input_tokens, 0) AS input_tokens,
            COALESCE(output_tokens, 0) AS output_tokens,
            COALESCE(cost_usd, 0.0) AS logged_cost,
            token_details
        FROM responses
        {where}
        ORDER BY datetime_utc
    """
    window_rows = list(db.execute(row_sql, params))

    # Batch-fetch prior-turn history for every conversation that has a
    # row in the window. One scan of ``responses`` filtered to those
    # conversation ids; we then index into the result per-row in Python.
    # History is drawn from the full responses table (not the window)
    # because the model saw the real history regardless of our filter.
    priors_by_conv: dict[str, list[tuple[str, str, str]]] = {}
    conv_ids = {r[2] for r in window_rows if r[2] is not None}
    if conv_ids:
        placeholders = ",".join("?" * len(conv_ids))
        for dt, conv, prompt, response in db.execute(
            f"""SELECT datetime_utc, conversation_id,
                       COALESCE(prompt, ''), COALESCE(response, '')
                FROM responses
                WHERE conversation_id IN ({placeholders})
                ORDER BY conversation_id, datetime_utc""",
            list(conv_ids),
        ):
            priors_by_conv.setdefault(conv, []).append((dt, prompt, response))

    attach_sigs = _attachment_signatures(db, [r[0] for r in window_rows])
    pfrag_sigs = _fragment_signatures(db, "prompt_fragments", [r[0] for r in window_rows])
    sfrag_sigs = _fragment_signatures(db, "system_fragments", [r[0] for r in window_rows])

    groups_by_fp: dict[str, list[tuple]] = {}
    for row in window_rows:
        (
            rid, dt, conv, model, resolved,
            system, prompt, options, schema,
            in_tok, out_tok, logged, token_details,
        ) = row
        prior_pairs = priors_by_conv.get(conv, ()) if conv is not None else ()
        prior_blob = "\x1f".join(
            f"{p}\x1e{r}" for pdt, p, r in prior_pairs if pdt < dt
        )
        fp_parts = (
            resolved or model,
            system,
            prompt,
            options,
            schema,
            prior_blob,
            attach_sigs.get(rid, ""),
            pfrag_sigs.get(rid, ""),
            sfrag_sigs.get(rid, ""),
        )
        fp = "\x1d".join(fp_parts)
        # Repack to the shape _row_cost expects: (_, _, model, resolved, _, in, out, logged, token_details)
        groups_by_fp.setdefault(fp, []).append(
            (rid, dt, model, resolved, None, in_tok, out_tok, logged, token_details)
        )

    per_model: dict[str, dict] = {}
    total_wasted = 0.0
    total_extra = 0
    dupe_group_count = 0
    for group in groups_by_fp.values():
        if len(group) < 2:
            continue
        dupe_group_count += 1
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
        total_groups=dupe_group_count,
        total_extra_calls=total_extra,
        total_wasted_usd=total_wasted,
        total_responses=total_count,
    )


def _row_cost(row: tuple, price: Price | None) -> float:
    """Cost for a single dupe-report row: logged if > 0 else priced."""
    _, _, _, _, _, inp, outp, logged, token_details = row
    logged_f = float(logged)
    if logged_f > 0:
        return logged_f
    if price is None:
        return 0.0
    breakdown = parse_token_details(token_details, int(inp), int(outp))
    return price.cost_for(breakdown)


def _attachment_signatures(
    db: sqlite_utils.Database, response_ids: list[str]
) -> dict[str, str]:
    """{response_id: '|'-joined attachment ids, ordered}. Empty when the
    ``prompt_attachments`` table isn't present on this logs.db."""
    if not response_ids or not db["prompt_attachments"].exists():
        return {}
    sigs: dict[str, list[str]] = {}
    for chunk in _chunked(response_ids, 500):
        placeholders = ",".join("?" * len(chunk))
        for rid, aid in db.execute(
            f"""SELECT response_id, attachment_id
                FROM prompt_attachments
                WHERE response_id IN ({placeholders})
                ORDER BY response_id, [order], attachment_id""",
            list(chunk),
        ):
            sigs.setdefault(rid, []).append(aid)
    return {rid: "|".join(ids) for rid, ids in sigs.items()}


def _fragment_signatures(
    db: sqlite_utils.Database, table: str, response_ids: list[str]
) -> dict[str, str]:
    """{response_id: '|'-joined fragment content hashes, ordered}. Keyed
    on ``fragments.hash`` so identical-content fragments fold together
    regardless of the local ``fragments.id`` surrogate."""
    if not response_ids or not db[table].exists() or not db["fragments"].exists():
        return {}
    sigs: dict[str, list[str]] = {}
    for chunk in _chunked(response_ids, 500):
        placeholders = ",".join("?" * len(chunk))
        for rid, h in db.execute(
            f"""SELECT t.response_id, fr.hash
                FROM {table} t
                JOIN fragments fr ON fr.id = t.fragment_id
                WHERE t.response_id IN ({placeholders})
                ORDER BY t.response_id, t.[order], fr.hash""",
            list(chunk),
        ):
            sigs.setdefault(rid, []).append(h)
    return {rid: "|".join(hs) for rid, hs in sigs.items()}


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


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
