"""Saisie interactive du chat avec `prompt_toolkit` : multiligne, Tab, historique, coloration.

Seul module qui connaît `prompt_toolkit` ; le presenter n'en reçoit que la fonction de lecture.
"""

import contextlib
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style

from iacli.domain.commands import InputKind, SlashCommand, parse_input
from iacli.domain.status import PromptStatus, context_level, format_tokens

STYLE = Style.from_dict(
    {
        "prompt": "bold ansigreen",
        "command": "bold ansicyan",
        "unknown-command": "ansired",
        "bottom-toolbar": "noreverse",
        "ok": "ansigreen",
        "warn": "ansiyellow",
        "danger": "bold ansired",
    }
)


class SlashCommandCompleter(Completer):
    """Complète le premier mot de la saisie quand il commence par `/`."""

    def __init__(self, commands: Sequence[SlashCommand]) -> None:
        self._commands = commands

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or any(char.isspace() for char in text):
            return
        for command in self._commands:
            if command.name.startswith(text.casefold()):
                yield Completion(
                    command.name,
                    start_position=-len(text),
                    display_meta=command.description,
                )


class SlashCommandLexer(Lexer):
    """Colore le premier mot de la saisie (première ligne non vide) quand le service le traiterait
    comme une commande.

    Cyan pour une commande valide ; rouge pour un mot inconnu ou une commande suivie de
    texte, que le service refuse. Le jugement porte sur le texte entier, comme à l'envoi."""

    def lex_document(self, document: Document) -> Callable[[int], StyleAndTextTuples]:
        lines = document.lines
        parsed = parse_input(document.text)
        # Le service analyse le premier mot du texte : celui de la première ligne non vide.
        first_line = next((i for i, line in enumerate(lines) if line.strip()), -1)

        def lex_line(line_number: int) -> StyleAndTextTuples:
            line = lines[line_number]
            if line_number != first_line or parsed.kind is InputKind.MESSAGE:
                return [("", line)]
            word, rest = _split_first_word(line)
            style = (
                "class:command"
                if parsed.kind is InputKind.COMMAND
                else "class:unknown-command"
            )
            tokens: StyleAndTextTuples = [(style, word)]
            if rest:
                tokens.append(("", rest))
            return tokens

        return lex_line


def _split_first_word(line: str) -> tuple[str, str]:
    stripped = line.lstrip()
    leading = len(line) - len(stripped)
    end = next((i for i, char in enumerate(stripped) if char.isspace()), len(stripped))
    return line[: leading + end], line[leading + end :]


def continuation_prompt(width: int, line_number: int, wrap_count: int) -> str:
    """Marge des lignes suivantes d'un message : des points, sur la largeur du prompt, pour que
    le texte reste aligné sous celui de la première ligne."""
    return "... ".ljust(width)[:width]


def render_toolbar(status: PromptStatus) -> StyleAndTextTuples:
    """Barre d'état : modèle et, une fois mesurée, occupation du contexte (colorée par niveau)."""
    tokens: StyleAndTextTuples = [("", f" modèle {status.model} ")]
    if status.context_used is not None and status.context_limit:
        style = f"class:{context_level(status.context_used, status.context_limit)}"
        used = format_tokens(status.context_used)
        limit = format_tokens(status.context_limit)
        tokens.append(("", "· contexte "))
        tokens.append(
            (
                style,
                f"{used} / {limit} tokens ({round(100 * status.context_used / status.context_limit)} %)",
            )
        )
        tokens.append(("", " "))
    return tokens


def resolve_enter(text: str) -> tuple[str, bool]:
    """Effet d'Entrée sur `text` : (nouveau texte, True pour envoyer le message).

    Les barres obliques inverses en fin de texte vont par paires : chaque paire donne une barre
    littérale ; si une barre reste seule, elle demande de continuer sur la ligne suivante."""
    backslash = "\\"
    trailing = len(text) - len(text.rstrip(backslash))
    body = text[: len(text) - trailing]
    literal = backslash * (trailing // 2)
    if trailing % 2:
        return body + literal + "\n", False
    return body + literal, True


def _key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit_or_continue(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        text, submit = resolve_enter(buffer.text)
        if text != buffer.text:
            buffer.text = text
            buffer.cursor_position = len(text)
        if submit:
            buffer.validate_and_handle()

    @bindings.add("c-j")
    @bindings.add("escape", "enter")
    def _new_line(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


class _BestEffortFileHistory(FileHistory):
    """Historique fichier qui ne fait jamais échouer la saisie : une erreur d'accès est ignorée."""

    def load_history_strings(self) -> Iterable[str]:
        try:
            yield from super().load_history_strings()
        except OSError:
            return

    def store_string(self, string: str) -> None:
        with contextlib.suppress(OSError):
            super().store_string(string)


def _create_history(history_path: Path | None) -> History:
    """Historique enregistré dans `history_path`, ou en mémoire (session seulement) sans chemin
    ou si le dossier est inutilisable."""
    if history_path is None:
        return InMemoryHistory()
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        _make_private(history_path)
    except OSError:
        return InMemoryHistory()
    return _BestEffortFileHistory(str(history_path))


def _make_private(path: Path) -> None:
    """Crée le fichier s'il manque et le réserve à son propriétaire (mode 0600) : il contient
    en clair ce que l'utilisateur a saisi. Sans effet sous Windows, où les droits du profil
    s'appliquent. Si le système de fichiers ignore les droits, l'historique reste enregistré."""
    os.close(os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600))
    with contextlib.suppress(OSError):
        path.chmod(0o600)


def create_prompt_input(
    commands: Sequence[SlashCommand],
    history_path: Path | None,
    status: Callable[[], PromptStatus],
    **session_options: Any,
) -> Callable[[str], str]:
    """Fonction de lecture du prompt : Entrée envoie ; une barre oblique inverse en fin
    de ligne, Ctrl+J ou Alt+Entrée passent à la ligne ; Tab complète les slash-commands et les messages sont historisés.

    Ctrl+D lève `EOFError` et Ctrl+C `KeyboardInterrupt`. `session_options` (entrée,
    sortie…) sont transmis à `PromptSession`, ce qui permet de tester sans terminal."""
    session: PromptSession[str] = PromptSession(
        history=_create_history(history_path),
        completer=SlashCommandCompleter(commands),
        lexer=SlashCommandLexer(),
        key_bindings=_key_bindings(),
        multiline=True,
        # Complétion à la demande (Tab) : pendant la frappe, elle empêche Tab d'insérer.
        complete_while_typing=False,
        prompt_continuation=continuation_prompt,
        bottom_toolbar=lambda: render_toolbar(status()),
        style=STYLE,
        **session_options,
    )

    def read(prompt: str) -> str:
        return session.prompt([("class:prompt", prompt)])

    return read
