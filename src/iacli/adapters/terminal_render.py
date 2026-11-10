"""Rendu riche pour terminal (markdown, diffs) : seul module qui connaît `rich`."""

import shutil
from io import StringIO

from rich.console import Console
from rich.markdown import Markdown


def _console(buffer: StringIO) -> Console:
    return Console(
        file=buffer,
        force_terminal=True,
        width=shutil.get_terminal_size().columns,
    )


def render_markdown(text: str) -> str:
    """Convertit du markdown en texte avec séquences ANSI (couleurs, gras, blocs de code)."""
    buffer = StringIO()
    _console(buffer).print(Markdown(text))
    return buffer.getvalue().rstrip("\n")
