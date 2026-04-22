from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
import sqlite_utils

from llm_cost.pricing import Price
from llm_cost.summary import (
    canonical_key,
    local_day_bounds,
    models_without_prices,
    summarise,
)


@pytest.fixture
def db(tmp_path):
    """A fresh logs.db with the columns llm-cost reads."""
    path = tmp_path / "logs.db"
    d = sqlite_utils.Database(path)
    d["responses"].create(
        {
            "id": str,
            "model": str,
            "resolved_model": str,
            "prompt": str,
            "system": str,
            "options_json": str,
            "schema_id": str,
            "conversation_id": str,
            "response": str,
            "input_tokens": int,
            "output_tokens": int,
            "cost_usd": float,
            "datetime_utc": str,
        },
        pk="id",
    )
    return d


def _insert(db, **row):
    row.setdefault("id", row["model"] + "-" + row["datetime_utc"])
    db["responses"].insert(row)


def test_summarise_empty_db(db):
    s = summarise(db)
    assert s.rows == ()
    assert s.total_cost_usd == 0.0


def test_summarise_aggregates_by_model(db):
    _insert(
        db,
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cost_usd=None,
        datetime_utc="2026-04-20T10:00:00+00:00",
    )
    _insert(
        db,
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        input_tokens=500_000,
        output_tokens=50_000,
        cost_usd=None,
        datetime_utc="2026-04-20T11:00:00+00:00",
    )

    s = summarise(db)
    assert len(s.rows) == 1
    row = s.rows[0]
    assert row.input_tokens == 1_500_000
    assert row.output_tokens == 150_000
    assert row.response_count == 2
    # Priced: 1.5M * $5 + 0.15M * $25
    assert row.priced_cost_usd == pytest.approx(1.5 * 5.0 + 0.15 * 25.0)
    assert row.priced is True


def test_summarise_prefers_logged_cost_usd_when_present(db):
    _insert(
        db,
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        input_tokens=1_000_000,
        output_tokens=0,
        cost_usd=4.20,  # llm's own logged cost — prefer this
        datetime_utc="2026-04-20T10:00:00+00:00",
    )
    s = summarise(db)
    row = s.rows[0]
    assert row.logged_cost_usd == 4.20
    # Priced column is still computed for audit
    assert row.priced_cost_usd == pytest.approx(5.0)
    # best_cost_usd picks logged
    assert row.best_cost_usd == 4.20
    assert s.total_cost_usd == 4.20


def test_summarise_flags_unpriced_models(db):
    _insert(
        db,
        model="made-up/frobnicator-v1",
        resolved_model="",
        input_tokens=1000,
        output_tokens=200,
        cost_usd=None,
        datetime_utc="2026-04-20T10:00:00+00:00",
    )
    s = summarise(db)
    row = s.rows[0]
    assert row.priced is False
    assert row.priced_cost_usd == 0.0
    assert s.has_unpriced is True
    assert list(models_without_prices(s)) == ["made-up/frobnicator-v1"]


def test_summarise_respects_time_window(db):
    for i, ts in enumerate(
        [
            "2026-04-19T23:00:00+00:00",
            "2026-04-20T01:00:00+00:00",
            "2026-04-20T23:00:00+00:00",
            "2026-04-21T01:00:00+00:00",
        ]
    ):
        _insert(
            db,
            id=f"r{i}",
            model="anthropic/claude-opus-4-6",
            resolved_model="claude-opus-4-6",
            input_tokens=1000,
            output_tokens=0,
            cost_usd=None,
            datetime_utc=ts,
        )
    since = datetime(2026, 4, 20, tzinfo=timezone.utc)
    until = datetime(2026, 4, 21, tzinfo=timezone.utc)
    s = summarise(db, since=since, until=until)
    assert s.rows[0].response_count == 2
    assert s.total_input == 2000


def test_summarise_applies_model_glob(db):
    _insert(
        db,
        id="keep",
        model="gemini/gemini-3-flash-preview",
        resolved_model="gemini-3-flash-preview",
        input_tokens=1000,
        output_tokens=100,
        cost_usd=None,
        datetime_utc="2026-04-20T00:00:00+00:00",
    )
    _insert(
        db,
        id="drop",
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        input_tokens=1000,
        output_tokens=100,
        cost_usd=None,
        datetime_utc="2026-04-20T00:00:00+00:00",
    )
    s = summarise(db, model_glob="gemini/%")
    assert len(s.rows) == 1
    # With no alias map, canonical_key strips the "gemini/" prefix.
    assert s.rows[0].model == "gemini-3-flash-preview"
    assert s.rows[0].variants == (("gemini/gemini-3-flash-preview", "gemini-3-flash-preview"),)


def test_summarise_collapses_variants_via_alias_map(db):
    """Three historical shapes of the same model roll up into one row."""
    for i, (model, resolved) in enumerate(
        [
            ("gemini-3-flash-preview", ""),                      # pre provider-prefix era
            ("gemini/gemini-3-flash-preview", ""),               # prefixed, no resolved
            ("gemini/gemini-3-flash-preview", "gemini-3-flash-preview"),  # prefixed + resolved
        ]
    ):
        _insert(
            db,
            id=f"r{i}",
            model=model,
            resolved_model=resolved,
            input_tokens=1000,
            output_tokens=100,
            cost_usd=None,
            datetime_utc=f"2026-04-20T0{i}:00:00+00:00",
        )
    # Alias map as llm would expose it: every alias (and the canonical id)
    # maps to the canonical model_id.
    alias_map = {
        "gemini/gemini-3-flash-preview": "gemini/gemini-3-flash-preview",
        "gemini-3-flash-preview": "gemini/gemini-3-flash-preview",
    }
    s = summarise(db, alias_map=alias_map)
    assert len(s.rows) == 1
    row = s.rows[0]
    assert row.model == "gemini/gemini-3-flash-preview"
    assert row.response_count == 3
    assert row.input_tokens == 3000
    assert len(row.variants) == 3
    # Pricing still resolves via _canonical inside pricing.resolve().
    assert row.priced is True


def test_summarise_without_alias_map_falls_back_to_heuristic(db):
    """Rows that only differ by prefix/resolved still collapse via _canonical."""
    _insert(
        db,
        id="a",
        model="gemini-3-flash-preview",
        resolved_model="",
        input_tokens=1000,
        output_tokens=0,
        cost_usd=None,
        datetime_utc="2026-04-20T00:00:00+00:00",
    )
    _insert(
        db,
        id="b",
        model="gemini/gemini-3-flash-preview",
        resolved_model="gemini-3-flash-preview",
        input_tokens=2000,
        output_tokens=0,
        cost_usd=None,
        datetime_utc="2026-04-20T01:00:00+00:00",
    )
    s = summarise(db)  # no alias map
    assert len(s.rows) == 1
    assert s.rows[0].model == "gemini-3-flash-preview"
    assert s.rows[0].input_tokens == 3000


def test_daily_summary_buckets_by_local_date(db):
    from llm_cost.summary import daily_summary

    # Two days of real spend, plus one outside the window.
    _insert(
        db,
        id="d1",
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        input_tokens=1_000_000,
        output_tokens=0,
        cost_usd=None,
        datetime_utc="2026-04-20T12:00:00+00:00",
    )
    _insert(
        db,
        id="d2",
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        input_tokens=2_000_000,
        output_tokens=0,
        cost_usd=None,
        datetime_utc="2026-04-21T12:00:00+00:00",
    )
    # Anchor "today" so the test is deterministic.
    rows = daily_summary(db, days=3, today=date(2026, 4, 21))
    # Exactly `days` rows, oldest first, empty days zero-filled.
    assert [r.day.isoformat() for r in rows] == ["2026-04-19", "2026-04-20", "2026-04-21"]
    assert rows[0].cost_usd == 0.0
    assert rows[1].cost_usd == pytest.approx(5.0)  # 1M * $5/1M
    assert rows[2].cost_usd == pytest.approx(10.0)  # 2M * $5/1M


def test_headlines_totals(db):
    from llm_cost.summary import headlines

    # Today, within this week, within this month.
    _insert(
        db,
        id="t",
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        input_tokens=1_000_000,
        output_tokens=0,
        cost_usd=None,
        datetime_utc="2026-04-21T12:00:00+00:00",
    )
    # Last month, still in all-time.
    _insert(
        db,
        id="old",
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        input_tokens=2_000_000,
        output_tokens=0,
        cost_usd=None,
        datetime_utc="2026-03-01T12:00:00+00:00",
    )
    head = headlines(db, today=date(2026, 4, 21))
    assert head.today == pytest.approx(5.0)
    assert head.this_week == pytest.approx(5.0)
    assert head.this_month == pytest.approx(5.0)  # month = April only
    assert head.all_time == pytest.approx(15.0)  # both rows
    # Top-3 is ordered by best cost (only one model here).
    assert len(head.top_models_month) == 1
    assert head.top_models_month[0].model == "claude-opus-4-6"


def test_top_responses_by_cost(db):
    from llm_cost.summary import top_responses

    # Small cheap model, big expensive model.
    _insert(
        db,
        id="cheap",
        model="gemini/gemini-2.5-flash-lite",
        resolved_model="",
        input_tokens=1000,
        output_tokens=100,
        cost_usd=None,
        datetime_utc="2026-04-20T10:00:00+00:00",
    )
    _insert(
        db,
        id="expensive",
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        input_tokens=1_000_000,
        output_tokens=50_000,
        cost_usd=None,
        datetime_utc="2026-04-20T11:00:00+00:00",
    )
    rows = top_responses(db, limit=5)
    assert len(rows) == 2
    assert rows[0].id == "expensive"
    assert rows[0].source == "priced"
    assert rows[0].cost_usd > rows[1].cost_usd


def test_top_responses_by_input_tokens(db):
    from llm_cost.summary import top_responses

    _insert(
        db,
        id="a",
        model="gemini/gemini-2.5-flash-lite",
        resolved_model="",
        input_tokens=500_000,
        output_tokens=100,
        cost_usd=None,
        datetime_utc="2026-04-20T10:00:00+00:00",
    )
    _insert(
        db,
        id="b",
        model="gemini/gemini-2.5-flash-lite",
        resolved_model="",
        input_tokens=200_000,
        output_tokens=100,
        cost_usd=None,
        datetime_utc="2026-04-20T11:00:00+00:00",
    )
    rows = top_responses(db, limit=2, by="input")
    assert [r.id for r in rows] == ["a", "b"]


def test_top_responses_prompt_preview_is_single_line(db):
    from llm_cost.summary import top_responses

    db["responses"].insert(
        {
            "id": "multi",
            "model": "anthropic/claude-opus-4-6",
            "resolved_model": "claude-opus-4-6",
            "prompt": "line one\n\nline two\n  tabbed\nline four",
            "input_tokens": 1000,
            "output_tokens": 100,
            "cost_usd": None,
            "datetime_utc": "2026-04-20T10:00:00+00:00",
        }
    )
    rows = top_responses(db, limit=1, preview_chars=80)
    assert rows[0].prompt_preview == "line one line two tabbed line four"


def _ensure_attachment_tables(db):
    if not db["attachments"].exists():
        db["attachments"].create({"id": str}, pk="id")
    if not db["prompt_attachments"].exists():
        db.execute(
            "CREATE TABLE prompt_attachments ("
            "response_id TEXT, attachment_id TEXT, [order] INTEGER,"
            "PRIMARY KEY (response_id, attachment_id))"
        )


def _ensure_fragment_tables(db):
    if not db["fragments"].exists():
        db["fragments"].create({"id": int, "hash": str}, pk="id")
    for name in ("prompt_fragments", "system_fragments"):
        if not db[name].exists():
            db.execute(
                f"CREATE TABLE {name} ("
                "response_id TEXT, fragment_id INTEGER, [order] INTEGER,"
                "PRIMARY KEY (response_id, fragment_id, [order]))"
            )


def _attach(db, response_id: str, attachment_id: str, order: int = 0):
    _ensure_attachment_tables(db)
    db["attachments"].insert({"id": attachment_id}, ignore=True)
    db["prompt_attachments"].insert(
        {"response_id": response_id, "attachment_id": attachment_id, "order": order}
    )


def _fragment(db, response_id: str, frag_id: int, hash_: str, order: int = 0, kind="prompt"):
    _ensure_fragment_tables(db)
    db["fragments"].insert({"id": frag_id, "hash": hash_}, ignore=True)
    db[f"{kind}_fragments"].insert(
        {"response_id": response_id, "fragment_id": frag_id, "order": order}
    )


def test_dupe_report_empty_db(db):
    from llm_cost.summary import dupe_report

    report = dupe_report(db)
    assert report.rows == ()
    assert report.total_responses == 0
    assert report.total_groups == 0


def test_dupe_report_identifies_waste(db):
    """Three identical prompts in distinct conversations → 2 extra calls."""
    from llm_cost.summary import dupe_report

    for i, ts in enumerate(
        ["2026-04-20T10:00:00+00:00", "2026-04-20T11:00:00+00:00", "2026-04-20T12:00:00+00:00"]
    ):
        _insert(
            db,
            id=f"dup{i}",
            model="anthropic/claude-opus-4-6",
            resolved_model="claude-opus-4-6",
            prompt="same prompt",
            conversation_id=f"c{i}",
            input_tokens=1_000_000,
            output_tokens=0,
            cost_usd=None,
            datetime_utc=ts,
        )
    _insert(
        db,
        id="unique",
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        prompt="different prompt",
        conversation_id="c-other",
        input_tokens=500_000,
        output_tokens=0,
        cost_usd=None,
        datetime_utc="2026-04-20T13:00:00+00:00",
    )

    report = dupe_report(db)
    assert report.total_groups == 1
    assert report.total_extra_calls == 2
    # Each call is 1M input * $5 = $5, so 2 wasted calls = $10
    assert report.total_wasted_usd == pytest.approx(10.0)
    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.model == "claude-opus-4-6"
    assert row.dupe_groups == 1
    assert row.extra_calls == 2


def test_dupe_report_system_prompt_breaks_fingerprint(db):
    """Same user prompt + different system prompt → not dupes."""
    from llm_cost.summary import dupe_report

    _insert(
        db,
        id="a",
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        prompt="hello",
        system="be terse",
        conversation_id="c1",
        datetime_utc="2026-04-20T10:00:00+00:00",
    )
    _insert(
        db,
        id="b",
        model="anthropic/claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        prompt="hello",
        system="be verbose",
        conversation_id="c2",
        datetime_utc="2026-04-20T11:00:00+00:00",
    )

    assert dupe_report(db).total_groups == 0


def test_dupe_report_conversation_history_breaks_fingerprint(db):
    """Same final prompt with different prior turns → not dupes."""
    from llm_cost.summary import dupe_report

    # Conversation 1: first turn "hi" → "hello", then "how are you"
    _insert(
        db, id="c1-t1", model="anthropic/claude-opus-4-6", resolved_model="claude-opus-4-6",
        prompt="hi", response="hello", conversation_id="c1",
        datetime_utc="2026-04-20T10:00:00+00:00",
    )
    _insert(
        db, id="c1-t2", model="anthropic/claude-opus-4-6", resolved_model="claude-opus-4-6",
        prompt="how are you", conversation_id="c1",
        datetime_utc="2026-04-20T10:01:00+00:00",
    )
    # Conversation 2: different prior turn, then the same "how are you"
    _insert(
        db, id="c2-t1", model="anthropic/claude-opus-4-6", resolved_model="claude-opus-4-6",
        prompt="yo", response="hey", conversation_id="c2",
        datetime_utc="2026-04-20T11:00:00+00:00",
    )
    _insert(
        db, id="c2-t2", model="anthropic/claude-opus-4-6", resolved_model="claude-opus-4-6",
        prompt="how are you", conversation_id="c2",
        datetime_utc="2026-04-20T11:01:00+00:00",
    )

    assert dupe_report(db).total_groups == 0


def test_dupe_report_attachments_break_fingerprint(db):
    """Same prompt + different attachment content → not dupes."""
    from llm_cost.summary import dupe_report

    _insert(
        db, id="a", model="anthropic/claude-opus-4-6", resolved_model="claude-opus-4-6",
        prompt="summarise this", conversation_id="c1",
        datetime_utc="2026-04-20T10:00:00+00:00",
    )
    _insert(
        db, id="b", model="anthropic/claude-opus-4-6", resolved_model="claude-opus-4-6",
        prompt="summarise this", conversation_id="c2",
        datetime_utc="2026-04-20T11:00:00+00:00",
    )
    _attach(db, "a", "hash-doc-A")
    _attach(db, "b", "hash-doc-B")

    assert dupe_report(db).total_groups == 0


def test_dupe_report_same_attachment_is_a_dupe(db):
    """Same prompt + same attachment content hash → dupe, even across conversations."""
    from llm_cost.summary import dupe_report

    _insert(
        db, id="a", model="anthropic/claude-opus-4-6", resolved_model="claude-opus-4-6",
        prompt="summarise this", conversation_id="c1",
        input_tokens=1_000_000, output_tokens=0,
        datetime_utc="2026-04-20T10:00:00+00:00",
    )
    _insert(
        db, id="b", model="anthropic/claude-opus-4-6", resolved_model="claude-opus-4-6",
        prompt="summarise this", conversation_id="c2",
        input_tokens=1_000_000, output_tokens=0,
        datetime_utc="2026-04-20T11:00:00+00:00",
    )
    _attach(db, "a", "shared-hash")
    _attach(db, "b", "shared-hash")

    report = dupe_report(db)
    assert report.total_groups == 1
    assert report.total_extra_calls == 1
    assert report.total_wasted_usd == pytest.approx(5.0)


def test_dupe_report_fragments_break_fingerprint(db):
    """Different system fragment content → not dupes."""
    from llm_cost.summary import dupe_report

    _insert(
        db, id="a", model="anthropic/claude-opus-4-6", resolved_model="claude-opus-4-6",
        prompt="go", conversation_id="c1",
        datetime_utc="2026-04-20T10:00:00+00:00",
    )
    _insert(
        db, id="b", model="anthropic/claude-opus-4-6", resolved_model="claude-opus-4-6",
        prompt="go", conversation_id="c2",
        datetime_utc="2026-04-20T11:00:00+00:00",
    )
    _fragment(db, "a", 1, "hash-frag-1", kind="system")
    _fragment(db, "b", 2, "hash-frag-2", kind="system")

    assert dupe_report(db).total_groups == 0


def test_canonical_key_prefers_alias_over_heuristic():
    amap = {"claude-haiku-4.5": "anthropic/claude-haiku-4-5-20251001"}
    # Resolved wins when it's in the alias map
    assert canonical_key("anything", "claude-haiku-4.5", amap) == (
        "anthropic/claude-haiku-4-5-20251001"
    )
    # Raw name wins when resolved isn't set
    assert canonical_key("claude-haiku-4.5", None, amap) == (
        "anthropic/claude-haiku-4-5-20251001"
    )
    # Falls back to the stripping heuristic when neither hits
    assert canonical_key("gemini/unknown-model", None, amap) == "unknown-model"


def test_summarise_honours_custom_prices(db):
    _insert(
        db,
        model="my-local-model",
        resolved_model="",
        input_tokens=1_000_000,
        output_tokens=0,
        cost_usd=None,
        datetime_utc="2026-04-20T00:00:00+00:00",
    )
    # Per-token costs: 0.01 / 1M input, 0.02 / 1M output (matches the
    # LiteLLM-aligned schema the loader now expects).
    s = summarise(db, prices={"my-local-model": Price(0.01 / 1_000_000, 0.02 / 1_000_000)})
    assert s.rows[0].priced_cost_usd == pytest.approx(0.01)


def test_local_day_bounds_brisbane_tz():
    """Brisbane is UTC+10, so 2026-04-22 local starts at 2026-04-21T14:00Z."""
    brisbane = ZoneInfo("Australia/Brisbane")
    start, end = local_day_bounds(date(2026, 4, 22), tz=brisbane)
    assert start == datetime(2026, 4, 21, 14, 0, tzinfo=timezone.utc)
    assert end == start + timedelta(days=1)
