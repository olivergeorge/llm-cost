"""llm-cost: token and spend reports over the llm logs database."""

from __future__ import annotations

import click
from llm import hookimpl

from .cli import register_commands as _register

__all__ = ["register_commands"]


@hookimpl
def register_commands(cli: click.Group) -> None:
    """Register the ``llm cost`` command group."""
    _register(cli)
