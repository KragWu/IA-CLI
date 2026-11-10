"""Saisie interactive du chat avec `prompt_toolkit` : multiligne, Tab, historique, coloration.

Seul module qui connaît `prompt_toolkit` ; le presenter n'en reçoit que la fonction de lecture.
"""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style

from iacli.adapters.cli_chat import PromptStatus
from iacli.domain.commands import SlashCommand, find_command, looks_like_command

# Part de la fenêtre de contexte à partir de laquelle la barre d'état passe à l'orange / au rouge.
WARN_RATIO = 0.6
DANGER_RATIO = 0.8

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
    """Colore le premier mot de la première ligne s'il a la forme d'une commande."""

    def __init__(self, commands: Sequence[SlashCommand]) -> None:
        self._commands = commands

    def lex_document(self, document: Document) -> Callable[[int], StyleAndTextTuples]:
        lines = document.lines

        def lex_line(line_number: int) -> StyleAndTextTuples:
            line = lines[line_number]
            if line_number != 0 or not looks_like_command(line):
                return [("", line)]
            word, rest = _split_first_word(line)
            style = (
                "class:command"
                if find_command(word) is not None
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


def render_toolbar(status: PromptStatus) -> StyleAndTextTuples:
    """Barre d'état : modèle et, une fois mesurée, occupation du contexte (colorée par niveau)."""
    tokens: StyleAndTextTuples = [("", f" modèle {status.model} ")]
    if status.context_used is not None and status.context_limit:
        ratio = status.context_used / status.context_limit
        style = (
            "class:danger"
            if ratio >= DANGER_RATIO
            else "class:warn"
            if ratio >= WARN_RATIO
            else "class:ok"
        )
        used = f"{status.context_used:,}".replace(",", " ")
        limit = f"{status.context_limit:,}".replace(",", " ")
        tokens.append(("", "· contexte "))
        tokens.append((style, f"{used} / {limit} tokens ({round(100 * ratio)} %)"))
        tokens.append(("", " "))
    return tokens


def _key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit_or_continue(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        if buffer.text.endswith("\\"):
            buffer.delete_before_cursor(1)
            buffer.insert_text("\n")
        else:
            buffer.validate_and_handle()

    @bindings.add("c-j")
    @bindings.add("escape", "enter")
    def _new_line(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


def create_prompt_input(
    commands: Sequence[SlashCommand],
    history_path: Path,
    status: Callable[[], PromptStatus],
    **session_options: Any,
) -> Callable[[str], str]:
    """Fonction de lecture du prompt : Entrée envoie ; une barre oblique inverse en fin
    de ligne, Ctrl+J ou Alt+Entrée passent à la ligne ; Tab complète les slash-commands et les messages sont historisés.

    Ctrl+D lève `EOFError` et Ctrl+C `KeyboardInterrupt`. `session_options` (entrée,
    sortie…) sont transmis à `PromptSession`, ce qui permet de tester sans terminal."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        completer=SlashCommandCompleter(commands),
        lexer=SlashCommandLexer(commands),
        key_bindings=_key_bindings(),
        multiline=True,
        # Complétion à la demande (Tab) : pendant la frappe, elle empêche Tab d'insérer.
        complete_while_typing=False,
        prompt_continuation="... ",
        bottom_toolbar=lambda: render_toolbar(status()),
        style=STYLE,
        **session_options,
    )

    def read(prompt: str) -> str:
        return session.prompt([("class:prompt", prompt)])

    return read
