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
    result = runner.invoke(cli, ["cost", "--db", str(seeded_db)])
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
    result = runner.invoke(cli, ["cost", "--db", str(seeded_db), "--json"])
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
    assert payload["rows"][0]["model"] == "gemini/gemini-3-flash-preview"


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
    prices.write_text(
        """
        made-up-model:
          input: 100.0
          output: 200.0
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
            "input_tokens": int,
            "output_tokens": int,
            "cost_usd": float,
            "datetime_utc": str,
        },
        pk="id",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "--db", str(empty)])
    assert result.exit_code == 0, result.output
    assert "No responses logged" in result.output
