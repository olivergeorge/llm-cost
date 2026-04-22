from __future__ import annotations

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from llm_cost import after_log_to_db, inline
from llm_cost.inline import format_cost_line


@pytest.fixture(autouse=True)
def reset_inline():
    """Keep test isolation — the ContextVar is process-global otherwise."""
    inline._ENABLED.set(None)
    yield
    inline._ENABLED.set(None)


def test_is_enabled_default_off(monkeypatch):
    monkeypatch.delenv("LLM_COST", raising=False)
    assert inline.is_enabled() is False


def test_is_enabled_env_override(monkeypatch):
    monkeypatch.setenv("LLM_COST", "1")
    assert inline.is_enabled() is True


def test_is_enabled_explicit_wins_over_env(monkeypatch):
    monkeypatch.setenv("LLM_COST", "1")
    inline.disable()
    assert inline.is_enabled() is False
    inline.enable()
    assert inline.is_enabled() is True


def test_format_cost_line_priced():
    # 1M input + 100k output * claude-opus-4-6 ($5 / $25 per 1M)
    alias_map = {"claude-opus-4-6": "anthropic/claude-opus-4-6"}
    line = format_cost_line(
        model_id="claude-opus-4-6",
        resolved_model="claude-opus-4-6",
        input_tokens=1_000_000,
        output_tokens=100_000,
        alias_map=alias_map,
    )
    # 1M * $5 + 0.1M * $25 = $7.50
    assert line == "Cost: $7.5000 (priced)"


def test_format_cost_line_unpriced():
    line = format_cost_line(
        model_id="made-up-model",
        resolved_model=None,
        input_tokens=1000,
        output_tokens=200,
        alias_map={},
    )
    assert line == "Cost: $0.0000 (unpriced)"


def test_after_log_to_db_skips_when_disabled(monkeypatch, capsys):
    monkeypatch.delenv("LLM_COST", raising=False)
    response = _fake_response("claude-opus-4-6", "claude-opus-4-6", 1_000_000, 100_000)
    after_log_to_db(response, db=None)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_after_log_to_db_emits_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("LLM_COST", "1")
    response = _fake_response("anthropic/claude-opus-4-6", "claude-opus-4-6", 1_000_000, 0)
    # Inject an alias map via monkeypatch so we don't depend on installed plugins.
    monkeypatch.setattr(inline, "_alias_map", lambda: {})
    after_log_to_db(response, db=None)
    captured = capsys.readouterr()
    # Anthropic prefix strips to claude-opus-4-6 via the heuristic path.
    assert "Cost: $5.0000 (priced)" in captured.err


def test_cost_flag_injected_into_prompt_command():
    """`llm prompt --cost ...` should be accepted by the injected option."""
    from llm_cost import register_commands

    @click.group()
    def root():
        pass

    @root.command()
    @click.argument("text", required=False)
    def prompt(text):
        click.echo(text or "no prompt")

    register_commands(root)

    runner = CliRunner()
    result = runner.invoke(root, ["prompt", "--cost", "hi"])
    assert result.exit_code == 0, result.output
    # The flag is absorbed (expose_value=False), prompt sees its own args.
    assert "hi" in result.output
    # And inline was enabled by the callback
    assert inline.is_enabled() is True


def test_no_cost_flag_overrides_env(monkeypatch):
    from llm_cost import register_commands

    monkeypatch.setenv("LLM_COST", "1")

    @click.group()
    def root():
        pass

    @root.command()
    def prompt():
        pass

    register_commands(root)

    runner = CliRunner()
    result = runner.invoke(root, ["prompt", "--no-cost"])
    assert result.exit_code == 0, result.output
    assert inline.is_enabled() is False


def _fake_response(model_id: str, resolved_model: str | None, inp: int, outp: int):
    """Minimal stand-in for llm.Response covering the attrs emit_cost reads."""
    return SimpleNamespace(
        prompt=SimpleNamespace(model=SimpleNamespace(model_id=model_id)),
        resolved_model=resolved_model,
        input_tokens=inp,
        output_tokens=outp,
    )


# click is imported at module level for the CLI tests above.
import click  # noqa: E402
