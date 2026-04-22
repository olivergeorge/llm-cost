"""llm-cost: token and spend reports over the llm logs database."""

from __future__ import annotations

__all__ = ["register_commands"]


def register_commands(cli):  # pragma: no cover - filled in by cli module
    from .cli import register_commands as _register

    _register(cli)
