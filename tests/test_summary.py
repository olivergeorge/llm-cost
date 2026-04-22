from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest
import sqlite_utils

from llm_cost.pricing import Price
from llm_cost.summary import (
    local_day_bounds,
    models_without_prices,
    summarise,
)


@pytest.fixture
def db(tmp_path):
    """A fresh logs.db with just the columns llm-cost reads."""
    path = tmp_path / "logs.db"
    d = sqlite_utils.Database(path)
    d["responses"].create(
        {
            "id": str,
            "model": str,
            "resolved_model": str,
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
    assert s.rows[0].model == "gemini/gemini-3-flash-preview"


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
    s = summarise(db, prices={"my-local-model": Price(0.01, 0.02)})
    assert s.rows[0].priced_cost_usd == pytest.approx(0.01)


def test_local_day_bounds_brisbane_tz():
    """Brisbane is UTC+10, so 2026-04-22 local starts at 2026-04-21T14:00Z."""
    brisbane = ZoneInfo("Australia/Brisbane")
    start, end = local_day_bounds(date(2026, 4, 22), tz=brisbane)
    assert start == datetime(2026, 4, 21, 14, 0, tzinfo=timezone.utc)
    assert end == start + timedelta(days=1)
