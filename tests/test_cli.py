from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
import sqlite_utils
from click.testing import CliRunner

from llm_cost import register_commands


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """A logs.db with three models spanning two days."""
    path = tmp_path / "logs.db"
    db = sqlite_utils.Database(path)
    db["responses"].create(
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
    db["responses"].insert_all(
        [
            {
                "id": "r1",
                "model": "anthropic/claude-opus-4-6",
                "resolved_model": "claude-opus-4-6",
                "input_tokens": 1_000_000,
                "output_tokens": 100_000,
                "cost_usd": None,
                "datetime_utc": "2026-04-20T10:00:00+00:00",
            },
            {
                "id": "r2",
                "model": "gemini/gemini-3-flash-preview",
                "resolved_model": "gemini-3-flash-preview",
                "input_tokens": 2_000_000,
                "output_tokens": 200_000,
                "cost_usd": 1.50,
                "datetime_utc": "2026-04-20T11:00:00+00:00",
            },
            {
                "id": "r3",
                "model": "made-up-model",
                "resolved_model": "",
                "input_tokens": 5_000,
                "output_tokens": 1_000,
                "cost_usd": None,
                "datetime_utc": "2026-04-19T09:00:00+00:00",
            },
        ]
    )
    return path


@pytest.fixture
def cli(seeded_db: Path):
    @click.group()
    def root():
        pass

    register_commands(root)
    return root


def test_cost_all_time_table(cli, seeded_db: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "all", "--db", str(seeded_db)])
    assert result.exit_code == 0, result.output
    # Every model row surfaces
    assert "claude-opus-4-6" in result.output
    assert "gemini-3-flash-preview" in result.output
    assert "made-up-model" in result.output
    # Source column marks logged / priced / unpriced
    assert "logged" in result.output
    assert "priced" in result.output
    assert "unpriced" in result.output
    # Unpriced footnote
    assert "Note: no price for:" in result.output
    assert "made-up-model" in result.output.split("Note:")[1]


def test_cost_json_output(cli, seeded_db: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "all", "--db", str(seeded_db), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["label"] == "all time"
    assert payload["total"]["responses"] == 3
    # Logged cost (1.50 for gemini) + priced cost (opus: 1M*$5 + 0.1M*$25 = $7.50)
    assert payload["total"]["cost_usd"] == pytest.approx(1.50 + 5.0 + 2.5)
    assert "made-up-model" in payload["unpriced_models"]


def test_cost_model_glob_filters(cli, seeded_db: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cost", "--db", str(seeded_db), "--model", "gemini/%", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["rows"]) == 1
    # Without an injected alias map (CLI test uses an empty click group),
    # canonical_key falls back to the prefix-stripping heuristic.
    assert payload["rows"][0]["model"] == "gemini-3-flash-preview"
    assert payload["rows"][0]["variants"] == [
        {"model": "gemini/gemini-3-flash-preview", "resolved_model": "gemini-3-flash-preview"}
    ]


def test_cost_since_until(cli, seeded_db: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "cost",
            "--db",
            str(seeded_db),
            "--since",
            "2026-04-20",
            "--until",
            "2026-04-20",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # Only the two 2026-04-20 rows; the 2026-04-19 made-up-model row is excluded.
    assert payload["total"]["responses"] == 2
    assert "made-up-model" not in {r["model"] for r in payload["rows"]}
    assert payload["label"] == "since 2026-04-20 until 2026-04-20"


def test_cost_prices_override(cli, seeded_db: Path, tmp_path: Path):
    prices = tmp_path / "override.yaml"
    # $100/M input, $200/M output expressed in the per-token units the
    # LiteLLM-aligned loader expects.
    prices.write_text(
        """
        made-up-model:
          input_cost_per_token: 100e-6
          output_cost_per_token: 200e-6
        """
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "cost",
            "--db",
            str(seeded_db),
            "--prices",
            str(prices),
            "--model",
            "made-up-model",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    row = payload["rows"][0]
    # 5000 * $100/1M + 1000 * $200/1M = $0.50 + $0.20 = $0.70
    assert row["priced_cost_usd"] == pytest.approx(0.7)
    assert row["priced"] is True


def test_cost_models_subcommand_lists_bundled_prices(cli):
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "models"])
    assert result.exit_code == 0, result.output
    assert "claude-opus-4-6" in result.output
    assert "gemini-3-flash-preview" in result.output
    # Header row
    assert "input $/1M" in result.output


def test_cost_empty_db(cli, tmp_path: Path):
    empty = tmp_path / "empty.db"
    db = sqlite_utils.Database(empty)
    db["responses"].create(
        {
            "id": str,
            "model": str,
            "resolved_model": str,
            "prompt": str,
            "input_tokens": int,
            "output_tokens": int,
            "cost_usd": float,
            "datetime_utc": str,
            "chain_hash": str,
            "replay_of": str,
        },
        pk="id",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "all", "--db", str(empty)])
    assert result.exit_code == 0, result.output
    assert "No responses logged" in result.output


def test_cost_default_daily_view(cli, seeded_db: Path):
    """Bare `llm cost` renders the landing page: sparkline + headlines."""
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "--db", str(seeded_db)])
    assert result.exit_code == 0, result.output
    assert "Spend — last 14 days" in result.output
    assert "Today" in result.output
    assert "This week" in result.output
    assert "This month" in result.output
    assert "All time" in result.output


def test_cost_top(cli, seeded_db: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "top", "--db", str(seeded_db), "-n", "2"])
    assert result.exit_code == 0, result.output
    assert "Top 2 expensive requests" in result.output
    # Two of the three seeded rows should appear — whichever the sort picks.
    # claude-opus-4-6 at 1M input tokens will be most expensive.
    assert "claude-opus-4-6" in result.output
    assert "llm-confirm-tokens" in result.output  # footer tip


def test_cost_top_json(cli, seeded_db: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "top", "--db", str(seeded_db), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sort_by"] == "cost"
    assert len(payload["rows"]) == 3  # three seeded responses
    # Highest cost first
    assert payload["rows"][0]["cost_usd"] >= payload["rows"][1]["cost_usd"]


def test_cost_dupes_no_dupes(cli, seeded_db: Path):
    """Seeded db has three distinct prompts — no dupes expected."""
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "dupes", "--db", str(seeded_db)])
    assert result.exit_code == 0, result.output
    assert "No duplicate requests detected" in result.output


def test_cost_dupes_finds_repeats(cli, tmp_path: Path):
    """Two identical prompts in different conversations → 1 extra call, $5 wasted."""
    path = tmp_path / "dupes.db"
    db = sqlite_utils.Database(path)
    db["responses"].create(
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
    for i, ts in enumerate(["2026-04-20T10:00:00+00:00", "2026-04-20T11:00:00+00:00"]):
        db["responses"].insert(
            {
                "id": f"r{i}",
                "model": "anthropic/claude-opus-4-6",
                "resolved_model": "claude-opus-4-6",
                "prompt": "explain recursion",
                "conversation_id": f"c{i}",
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "cost_usd": None,
                "datetime_utc": ts,
            }
        )

    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "dupes", "--db", str(path)])
    assert result.exit_code == 0, result.output
    assert "Dupe spend" in result.output
    assert "claude-opus-4-6" in result.output
    # 1 extra call @ $5/1M * 1M = $5 wasted
    assert "$5.00" in result.output


def test_cost_default_daily_json(cli, seeded_db: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "--db", str(seeded_db), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["days"]) == 14
    assert "headlines" in payload
    assert set(payload["headlines"]) == {
        "today", "this_week", "this_month", "all_time", "top_models_month"
    }


class TestFormatMoney:
    """Dollars-and-cents for aggregates, sub-cent precision for tiny per-request costs."""

    def test_cents_above_a_penny(self):
        from llm_cost.cli import format_money

        assert format_money(8.0463) == "$8.05"
        assert format_money(0.15) == "$0.15"
        assert format_money(1_234.5) == "$1,234.50"

    def test_sub_cent_keeps_four_decimals(self):
        from llm_cost.cli import format_money

        # A short prompt on a cheap model legitimately costs this much;
        # 2dp would collapse it to "$0.00" which reads as free.
        assert format_money(0.0017) == "$0.0017"
        assert format_money(0.003) == "$0.0030"

    def test_exact_zero_uses_cents(self):
        from llm_cost.cli import format_money

        # Unpriced model → $0 exactly → display as $0.00 (not $0.0000),
        # since zero has no precision to lose and the aggregate format
        # is the least surprising.
        assert format_money(0) == "$0.00"

    def test_boundary_just_below_a_penny(self):
        from llm_cost.cli import format_money

        # $0.0099 rounds to $0.01 at 2dp → belongs in the cents bucket.
        assert format_money(0.0099) == "$0.01"
        # $0.0049 rounds to $0.00 at 2dp → keep four decimals so it's
        # not indistinguishable from exact zero.
        assert format_money(0.0049) == "$0.0049"
