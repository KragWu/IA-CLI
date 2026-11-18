"""Rendu riche pour terminal (markdown, diffs) : seul module qui connaît `rich`."""

import shutil
from io import StringIO

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

# Largeur minimale de rendu : en dessous, le markdown devient illisible.
MIN_WIDTH = 20


def supports_ansi(console: Console | None = None) -> bool:
    """True si la sortie est un terminal capable d'afficher des séquences ANSI.

    Une console Windows historique (sans séquences ANSI) afficherait les codes en clair."""
    console = console or Console()
    return console.is_terminal and not console.legacy_windows


def _console(buffer: StringIO, margin: int = 0) -> Console:
    width = max(MIN_WIDTH, shutil.get_terminal_size().columns - margin)
    return Console(file=buffer, force_terminal=True, width=width)


def render_markdown(text: str, margin: int = 0) -> str:
    """Convertit du markdown en texte avec séquences ANSI (couleurs, gras, blocs de code).

    `margin` est le nombre de colonnes déjà occupées à gauche (préfixe, retrait)."""
    buffer = StringIO()
    _console(buffer, margin).print(Markdown(text))
    return buffer.getvalue().rstrip("\n")


def render_diff(diff: str) -> str:
    """Colore un diff unifié : lignes ajoutées, supprimées et en-têtes de bloc."""
    buffer = StringIO()
    syntax = Syntax(diff, "diff", theme="ansi_dark", background_color="default")
    _console(buffer).print(syntax)
    return buffer.getvalue().rstrip("\n")
