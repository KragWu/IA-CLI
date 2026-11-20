"""Adapter Textual du chat (option `--tui`) : presenter et application en mode en ligne.

Seul module qui connaît `textual`. Le presenter reçoit une fonction `emit` sûre depuis
n'importe quel thread (l'application la fournit avec `call_from_thread`)."""

import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import ModalScreen
from textual.widgets import Input, Label, RichLog

from iacli.domain.commands import SlashCommand
from iacli.domain.status import format_duration, format_tokens
from iacli.domain.text import neutralize_control_characters as safe
from iacli.services.chat import ChatSession

RESPONSE_PREFIX = "iacli> "


class TextualPresenter:
    """Implémente le port `ChatPresenter` pour l'interface Textual.

    Chaque affichage est une ligne envoyée à `emit`. `render_markdown` (optionnel) transforme
    la réponse en texte avec séquences ANSI, que l'application affiche en style, et
    `render_diff` colore les diffs de la même façon. `ask` pose une question fermée à
    l'utilisateur (bloquant, appelé depuis le thread de travail) ; sans lui, les écritures
    et les commandes sont refusées en le disant."""

    def __init__(
        self,
        emit: Callable[[str], None],
        render_markdown: Callable[[str], str] | None = None,
        render_diff: Callable[[str], str] | None = None,
        ask: Callable[[str], bool] | None = None,
    ) -> None:
        self._emit = emit
        self._render_markdown = render_markdown
        self._render_diff = render_diff
        self._ask = ask
        self.context_usage: tuple[int, int] | None = None

    def start(self, workspace: Path, model: str) -> None:
        self._emit(f"IA-CLI — modèle {safe(model)} — projet {safe(str(workspace))}")
        self._emit("Entrez /help pour les commandes, /exit ou /quit pour terminer.")

    def show_response(self, response: str) -> None:
        response = safe(response)
        if self._render_markdown is not None:
            try:
                rendered = self._render_markdown(response)
            except Exception:
                rendered = (
                    response  # rendu impossible : le texte brut vaut mieux que rien
                )
            # Les lignes suivantes s'alignent sous le texte, après le préfixe.
            response = rendered.replace("\n", "\n" + " " * len(RESPONSE_PREFIX))
        self._emit(f"\n{RESPONSE_PREFIX}{response}\n")

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
        diff = safe(diff)
        rendered = self._render_diff_or_plain(diff)
        self._emit(f"Modification proposée pour {safe(path)} :")
        self._emit(rendered)
        return self._confirm(
            "Appliquer cette modification ? [o/N]",
            refused=f"modification de {safe(path)} refusée.",
        )

    def confirm_shell_command(self, command: str, workspace: Path) -> bool:
        self._emit(f"Commande proposée dans {safe(str(workspace))} : {safe(command)}")
        return self._confirm(
            "Exécuter cette commande ? [o/N]",
            refused=f"commande {safe(command)} refusée.",
        )

    def _render_diff_or_plain(self, diff: str) -> str:
        if self._render_diff is None:
            return diff
        try:
            return self._render_diff(diff)
        except Exception:
            return diff  # rendu impossible : le diff brut vaut mieux que rien

    def _confirm(self, question: str, refused: str) -> bool:
        """Pose la question ; sans moyen de la poser, l'action est refusée (et on le dit)."""
        if self._ask is None:
            self._emit(
                f"erreur> confirmation indisponible dans cette interface : {refused}"
            )
            return False
        return self._ask(question)


class ConfirmationScreen(ModalScreen[bool]):
    """Question fermée posée pendant qu'un outil attend : `o` ou `y` accepte, tout le reste
    refuse (refus par défaut, comme la console)."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("o,y", "answer(True)", "Oui"),
        Binding("n,enter,escape", "answer(False)", "Non"),
    ]

    def __init__(self, question: str) -> None:
        super().__init__(id="confirmation")
        self._question = question

    def compose(self) -> ComposeResult:
        yield Label(safe(self._question))

    def action_answer(self, accepted: bool) -> None:
        self.dismiss(accepted)


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
        self,
        create_session: Callable[[TextualPresenter], ChatSession],
        render_markdown: Callable[[str], str] | None = None,
        render_diff: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__()
        self._confirmation: ConfirmationScreen | None = None
        self.transcript: list[str] = []
        self._busy = False
        self._ui_thread = threading.get_ident()
        self._presenter = TextualPresenter(
            emit=self._emit,
            render_markdown=render_markdown,
            render_diff=render_diff,
            ask=self._ask,
        )
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
        # Seul le rendu markdown produit des séquences ANSI : le texte venu du modèle en est
        # débarrassé par `safe` avant d'y arriver.
        self.query_one("#log", RichLog).write(
            Text.from_ansi(line) if chr(27) in line else line
        )

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

    def _ask(self, question: str) -> bool:
        """Pose la question et attend la réponse ; appelé depuis le thread de travail."""
        answered = threading.Event()
        answer: list[bool] = []

        def finish(accepted: bool | None) -> None:
            answer.append(bool(accepted))
            answered.set()

        self.call_from_thread(self._open_confirmation, question, finish)
        answered.wait()
        return answer[0]

    def _open_confirmation(
        self, question: str, finish: Callable[[bool | None], None]
    ) -> None:
        screen = ConfirmationScreen(question)
        self._confirmation = screen
        self.push_screen(screen, finish)

    def _refuse_confirmation(self) -> None:
        if self._confirmation is not None and self._confirmation.is_attached:
            self._confirmation.dismiss(False)
        self._confirmation = None

    def action_interrupt(self) -> None:
        self._refuse_confirmation()
        if self._busy:
            self._session.cancel()
        else:
            self.exit()

    def action_leave(self) -> None:
        self._refuse_confirmation()
        self._session.cancel()
        self.exit()
