"""Adapter Textual du chat (option `--tui`) : presenter et application en mode en ligne.

Seul module qui connaît `textual`. Le presenter reçoit une fonction `emit` sûre depuis
n'importe quel thread (l'application la fournit avec `call_from_thread`)."""

import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import Input, RichLog

from iacli.domain.commands import SlashCommand
from iacli.domain.status import format_duration, format_tokens
from iacli.domain.text import neutralize_control_characters as safe
from iacli.services.chat import ChatSession

RESPONSE_PREFIX = "iacli> "


class TextualPresenter:
    """Implémente le port `ChatPresenter` pour l'interface Textual.

    Chaque affichage est une ligne envoyée à `emit`. Les confirmations ne sont pas encore
    disponibles : elles refusent l'action en le disant."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self.context_usage: tuple[int, int] | None = None

    def start(self, workspace: Path, model: str) -> None:
        self._emit(f"IA-CLI — modèle {safe(model)} — projet {safe(str(workspace))}")
        self._emit("Entrez /help pour les commandes, /exit ou /quit pour terminer.")

    def show_response(self, response: str) -> None:
        self._emit(f"\n{RESPONSE_PREFIX}{safe(response)}\n")

    def show_error(self, message: str) -> None:
        self._emit(f"erreur> {safe(message)}")

    def show_notice(self, message: str) -> None:
        self._emit(safe(message))

    def show_help(self, commands: Sequence[SlashCommand]) -> None:
        self._emit("Commandes disponibles :")
        for command in commands:
            self._emit(f"  {command.name:<10} {command.description}")

    def show_cancelled(self) -> None:
        self._emit("Génération annulée")

    def show_tool_call(self, name: str, arguments: dict[str, object]) -> None:
        serialized = json.dumps(arguments, ensure_ascii=False)
        self._emit(f"outil> {safe(name)}({safe(serialized)})")

    def show_tool_result(self, name: str) -> None:
        self._emit(f"outil terminé> {safe(name)}")

    def show_action_timing(self, action: str, seconds: float) -> None:
        self._emit(f"temps action> {safe(action)} : {format_duration(seconds)}")

    def show_total_duration(self, seconds: float) -> None:
        self._emit(f"temps total> demande : {format_duration(seconds)}")

    def show_context_usage(self, used: int, limit: int) -> None:
        self.context_usage = (used, limit)
        self._emit(
            f"contexte> {format_tokens(used)} / {format_tokens(limit)} "
            f"tokens ({round(100 * used / limit)} %)"
        )

    def clear_context_usage(self) -> None:
        self.context_usage = None

    def show_context_warning(self, used: int, limit: int) -> None:
        self._emit(
            f"attention> contexte utilisé à {round(100 * used / limit)} % : au-delà de "
            f"{format_tokens(limit)} tokens, Ollama peut tronquer les plus "
            "anciens messages de la conversation."
        )

    def start_progress(self) -> None:
        self._emit("L'agent traite votre demande...")

    def update_progress(self, message: str) -> None:
        """Pas encore d'indicateur animé (étape suivante) : le message est ignoré."""

    def stop_progress(self) -> None:
        """Pas encore d'indicateur animé (étape suivante)."""

    def confirm_file_write(self, path: str, diff: str) -> bool:
        self._emit(
            f"erreur> confirmation indisponible dans cette interface : "
            f"modification de {safe(path)} refusée."
        )
        return False

    def confirm_shell_command(self, command: str, workspace: Path) -> bool:
        self._emit(
            "erreur> confirmation indisponible dans cette interface : "
            f"commande {safe(command)} refusée."
        )
        return False


class ChatApp(App[None]):
    """Conversation dans le terminal, en mode en ligne.

    `ChatSession.handle` tourne dans un thread de travail : l'interface reste réactive et le
    presenter lui renvoie chaque ligne par `call_from_thread`. Ctrl+C annule le tour en cours,
    et quitte immédiatement quand aucun tour n'est en cours ; Ctrl+D quitte toujours."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "interrupt", "Annuler ou quitter", priority=True),
        Binding("ctrl+d", "leave", "Quitter", priority=True),
    ]

    def __init__(
        self, create_session: Callable[[TextualPresenter], ChatSession]
    ) -> None:
        super().__init__()
        self.transcript: list[str] = []
        self._busy = False
        self._ui_thread = threading.get_ident()
        self._presenter = TextualPresenter(emit=self._emit)
        self._session = create_session(self._presenter)

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        yield Input(id="prompt", placeholder="vous> ")

    def on_mount(self) -> None:
        self._ui_thread = threading.get_ident()
        self._session.start()
        self.query_one("#prompt", Input).focus()

    def _emit(self, line: str) -> None:
        """Ajoute une ligne au journal, depuis n'importe quel thread."""
        if threading.get_ident() == self._ui_thread:
            self._write(line)
        else:
            self.call_from_thread(self._write, line)

    def _write(self, line: str) -> None:
        self.transcript.append(line)
        self.query_one("#log", RichLog).write(line)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value
        event.input.value = ""
        if self._busy or not text.strip():
            return
        self._busy = True
        event.input.disabled = True
        self._write(f"vous> {safe(text)}")
        self.run_worker(lambda: self._handle(text), thread=True)

    def _handle(self, text: str) -> None:
        keep_going = self._session.handle(text)
        self.call_from_thread(self._turn_finished, keep_going)

    def _turn_finished(self, keep_going: bool) -> None:
        self._busy = False
        if not keep_going:
            self.exit()
            return
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()

    def action_interrupt(self) -> None:
        if self._busy:
            self._session.cancel()
        else:
            self.exit()

    def action_leave(self) -> None:
        self._session.cancel()
        self.exit()
