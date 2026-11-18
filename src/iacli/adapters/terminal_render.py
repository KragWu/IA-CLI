"""Rendu riche pour terminal (markdown, diffs) : seul module qui connaît `rich`."""

import shutil
from io import StringIO

from markdown_it import MarkdownIt
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

# Largeur minimale de rendu : en dessous, le markdown devient illisible.
MIN_WIDTH = 20


def supports_ansi(console: Console | None = None) -> bool:
    """True si la sortie est un terminal capable d'afficher des séquences ANSI.

    Une console Windows historique ou un terminal `TERM=dumb` afficherait les codes en clair."""
    console = console or Console()
    return (
        console.is_terminal
        and not console.legacy_windows
        and not console.is_dumb_terminal
    )


def _console(buffer: StringIO, margin: int = 0) -> Console:
    width = max(MIN_WIDTH, shutil.get_terminal_size().columns - margin)
    return Console(file=buffer, force_terminal=True, width=width)


class _TextMarkdown(Markdown):
    """Markdown où le HTML est du texte : `List<String>` ou `<div>` s'affichent tels que le
    modèle les a écrits au lieu d'être avalés comme des balises."""

    def __init__(self, markup: str) -> None:
        # `Markdown.__init__` analyse toujours son texte avec un parseur qui interprète le HTML :
        # on le laisse analyser du vide, puis on installe notre propre analyse (`markup` et
        # `parsed` sont les attributs que `rich` relit au rendu).
        super().__init__("")
        parser = MarkdownIt("commonmark", {"html": False})
        self.markup = markup
        self.parsed = parser.enable("strikethrough").enable("table").parse(markup)


def render_markdown(text: str, margin: int = 0) -> str:
    """Convertit du markdown en texte avec séquences ANSI (couleurs, gras, blocs de code).

    `margin` est le nombre de colonnes déjà occupées à gauche (préfixe, retrait)."""
    buffer = StringIO()
    _console(buffer, margin).print(_TextMarkdown(text))
    return buffer.getvalue().rstrip("\n")


def render_diff(diff: str) -> str:
    """Colore un diff unifié : lignes ajoutées, supprimées et en-têtes de bloc."""
    buffer = StringIO()
    syntax = Syntax(diff, "diff", theme="ansi_dark", background_color="default")
    _console(buffer).print(syntax)
    return buffer.getvalue().rstrip("\n")
