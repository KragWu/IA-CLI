"""Adapter Textual du chat (option `--tui`) : presenter et application en mode en ligne.

Seul module qui connaît `textual`. Le presenter reçoit une fonction `emit` sûre depuis
n'importe quel thread (l'application la fournit avec `call_from_thread`)."""

import contextlib
import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar, Protocol

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.message import Message
from textual.widgets import Label, RichLog, TextArea

from iacli.domain.commands import (
    COMMANDS,
    SlashCommand,
    common_prefix,
    matching_commands,
    resolve_enter,
)
from iacli.domain.history import InputHistory
from iacli.domain.status import (
    context_level,
    context_percent,
    format_duration,
    format_tokens,
)
from iacli.domain.text import neutralize_control_characters as safe
from iacli.services.chat import ChatSession
from iacli.services.ports import HistoryStore

RESPONSE_PREFIX = "iacli> "


class ChatScreen(Protocol):
    """Ce que le presenter pilote dans l'interface en dehors du journal : indicateur de
    progression animé et barre d'état. Appelé depuis n'importe quel thread."""

    def start_progress(self) -> None: ...

    def set_progress_message(self, message: str) -> None: ...

    def pause_progress(self) -> None: ...

    def resume_progress(self) -> None: ...

    def stop_progress(self) -> None: ...

    def set_model(self, model: str) -> None: ...

    def set_context(self, context: tuple[int, int] | None) -> None: ...


class TextualPresenter:
    """Implémente le port `ChatPresenter` pour l'interface Textual.

    Chaque affichage est une ligne envoyée à `emit`. `render_markdown` (optionnel) transforme
    la réponse en texte avec séquences ANSI, que l'application affiche en style, et
    `render_diff` colore les diffs de la même façon. `ask` pose une question fermée à
    l'utilisateur (bloquant, appelé depuis le thread de travail) ; sans lui, les écritures
    et les commandes sont refusées en le disant. `screen` reçoit la progression et le niveau
    de contexte pour l'indicateur animé et la barre d'état."""

    def __init__(
        self,
        emit: Callable[[str], None],
        render_markdown: Callable[[str], str] | None = None,
        render_diff: Callable[[str], str] | None = None,
        ask: Callable[[str], bool] | None = None,
        screen: "ChatScreen | None" = None,
    ) -> None:
        self._emit = emit
        self._screen = screen
        self._render_markdown = render_markdown
        self._render_diff = render_diff
        self._ask = ask
        self.context_usage: tuple[int, int] | None = None

    def start(self, workspace: Path, model: str) -> None:
        if self._screen is not None:
            self._screen.set_model(safe(model))
        self._emit(f"IA-CLI — modèle {safe(model)} — projet {safe(str(workspace))}")
        self._emit("Entrez /help pour les commandes, /exit ou /quit pour terminer.")

    def show_response(self, response: str) -> None:
        response = safe(response)
        if self._render_markdown is not None:
            # Rendu impossible : le texte brut vaut mieux que rien.
            with contextlib.suppress(Exception):
                response = self._render_markdown(response)
        # Les lignes suivantes s'alignent sous le texte, après le préfixe.
        response = response.replace("\n", "\n" + " " * len(RESPONSE_PREFIX))
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
        if self._screen is not None:
            self._screen.set_context((used, limit))
        self._emit(
            f"contexte> {format_tokens(used)} / {format_tokens(limit)} "
            f"tokens ({context_percent(used, limit)} %)"
        )

    def clear_context_usage(self) -> None:
        self.context_usage = None
        if self._screen is not None:
            self._screen.set_context(None)

    def show_context_warning(self, used: int, limit: int) -> None:
        self._emit(
            f"attention> contexte utilisé à {context_percent(used, limit)} % : au-delà de "
            f"{format_tokens(limit)} tokens, Ollama peut tronquer les plus "
            "anciens messages de la conversation."
        )

    def start_progress(self) -> None:
        if self._screen is None:
            self._emit("L'agent traite votre demande...")
        else:  # l'indicateur animé affiche déjà ce message
            self._screen.start_progress()

    def update_progress(self, message: str) -> None:
        if self._screen is not None:
            self._screen.set_progress_message(safe(message))

    def stop_progress(self) -> None:
        if self._screen is not None:
            self._screen.stop_progress()

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
        if self._screen is not None:
            self._screen.pause_progress()
        try:
            return self._ask(question)
        finally:
            if self._screen is not None:
                self._screen.resume_progress()


class PromptArea(TextArea):
    """Saisie multiligne. Entrée envoie (une barre oblique inverse en fin de ligne continue sur
    la suivante) ; Ctrl+J ou Maj+Entrée passent à la ligne ; Tab complète les slash-commands ;
    les flèches haut et bas parcourent l'historique depuis la première et la dernière ligne."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, *, id: str, placeholder: str) -> None:
        super().__init__(
            id=id,
            placeholder=placeholder,
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
            highlight_cursor_line=False,
        )
        self.input_history = InputHistory()

    def _show_hint(self, text: str) -> None:
        hint = self.app.query_one("#hint", Label)
        hint.update(Text(text))
        hint.display = bool(text)

    def _replace(self, text: str) -> None:
        self.text = text
        self.move_cursor(self.document.end)

    async def _on_key(self, event: events.Key) -> None:
        key = event.key
        if key != "tab":
            self._show_hint("")
        if key == "enter":
            self._handled(event)
            text, submit = resolve_enter(self.text)
            if text != self.text:
                self._replace(text)
            if submit:
                self.post_message(self.Submitted(self.text))
        elif key in {"ctrl+j", "shift+enter"}:
            self._handled(event)
            self.insert("\n")
        elif key == "tab":
            self._handled(event)
            self._complete()
        elif key == "up" and self.navigator.is_first_wrapped_line(self.cursor_location):
            self._handled(event)
            self._browse(self.input_history.previous(self.text))
        elif key == "down" and self.navigator.is_last_wrapped_line(
            self.cursor_location
        ):
            self._handled(event)
            self._browse(self.input_history.next())

    @staticmethod
    def _handled(event: events.Key) -> None:
        event.stop()
        event.prevent_default()

    def _browse(self, text: str | None) -> None:
        if text is not None:
            self._replace(text)

    def _complete(self) -> None:
        matches = matching_commands(self.text, COMMANDS)
        if not matches:
            return
        names = [command.name for command in matches]
        self._replace(common_prefix(names))
        self._show_hint("  ".join(names) if len(names) > 1 else "")


class ChatApp(App[None]):
    """Conversation dans le terminal, en mode en ligne.

    `ChatSession.handle` tourne dans un thread de travail : l'interface reste réactive et le
    presenter lui renvoie chaque ligne par `call_from_thread`. Ctrl+C annule le tour en cours,
    et quitte immédiatement quand aucun tour n'est en cours ; Ctrl+D quitte toujours."""

    CSS = """
    #question { display: none; text-style: bold; color: yellow; }
    #progress { display: none; }
    #hint { display: none; color: $text-muted; }
    #prompt { height: auto; max-height: 8; border: none; padding: 0; }
    #status { background: $panel; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "interrupt", "Annuler ou quitter", priority=True),
        Binding("ctrl+d", "leave", "Quitter", priority=True),
    ]

    def __init__(
        self,
        create_session: Callable[[TextualPresenter], ChatSession],
        render_markdown: Callable[[str], str] | None = None,
        render_diff: Callable[[str], str] | None = None,
        history: HistoryStore | None = None,
    ) -> None:
        super().__init__()
        self._history_store = history
        self._turn_thread: threading.Thread | None = None
        self._cancel_requested = False
        self._turn_id = 0
        self._pending_answer: Callable[[bool], None] | None = None
        self._model_name = ""
        self._context_usage: tuple[int, int] | None = None
        self._progress_active = False
        self._progress_paused = False
        self._progress_message = ""
        self._progress_started = 0.0
        self._progress_frame = 0
        self.transcript: list[str] = []
        self._busy = False
        self._ui_thread = threading.get_ident()
        self._presenter = TextualPresenter(
            emit=self._emit,
            render_markdown=render_markdown,
            render_diff=render_diff,
            ask=self._ask,
            screen=self,
        )
        self._session = create_session(self._presenter)

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        yield Label(id="progress")
        yield Label(id="question")
        yield Label(id="hint")
        yield PromptArea(id="prompt", placeholder="vous> ")
        yield Label(id="status")

    def on_mount(self) -> None:
        self._ui_thread = threading.get_ident()
        self.set_interval(0.1, self._advance_spinner)
        self._session.start()
        self._draw_status()
        prompt = self.query_one("#prompt", PromptArea)
        if self._history_store is not None:
            prompt.input_history = InputHistory(self._history_store.load())
        prompt.focus()

    def _emit(self, line: str) -> None:
        """Ajoute une ligne au journal, depuis n'importe quel thread."""
        if threading.get_ident() == self._ui_thread:
            self._write(line)
        else:
            self._call_ui(self._write, line)

    def _write(self, line: str) -> None:
        self.transcript.append(line)
        # Seul le rendu markdown produit des séquences ANSI : le texte venu du modèle en est
        # débarrassé par `safe` avant d'y arriver.
        self.query_one("#log", RichLog).write(
            Text.from_ansi(line) if chr(27) in line else line
        )

    def on_prompt_area_submitted(self, event: PromptArea.Submitted) -> None:
        text = event.text
        prompt = self.query_one("#prompt", PromptArea)
        prompt.clear()
        if self._busy or not text.strip():
            return
        if prompt.input_history.add(text) and self._history_store is not None:
            self._history_store.append(text)
        self._busy = True
        self._turn_id += 1
        prompt.disabled = True
        self._write(f"vous> {safe(text)}")
        # Thread daemon : un appel du modèle en cours ne retient pas la fermeture de l'application
        # (un worker Textual, lui, la ferait attendre la fin de l'appel).
        self._turn_thread = threading.Thread(
            target=self._handle, args=(text,), daemon=True
        )
        self._turn_thread.start()

    def _call_ui(self, function: Callable[..., Any], *args: Any) -> bool:
        """Exécute `function` dans le thread de l'interface ; False si l'application est fermée
        (le tour qui finit après la sortie n'a plus rien à afficher)."""
        if not self.is_running:
            return False
        try:
            self.call_from_thread(function, *args)
        except RuntimeError:
            # `call_from_thread` relève aussi les erreurs du callback : seule la fermeture de
            # l'application entre le test et l'appel est une absence de réponse normale.
            if self.is_running:
                raise
            return False
        return True

    def _handle(self, text: str) -> None:
        keep_going = True
        try:
            keep_going = self._session.handle(text)
        except Exception as error:
            # Défaillance que la session n'a pas traitée : on la montre et on rend la main,
            # sinon la saisie resterait désactivée.
            self._presenter.show_error(str(error) or type(error).__name__)
        finally:
            self._call_ui(self._turn_finished, keep_going)

    def _turn_finished(self, keep_going: bool) -> None:
        self._busy = False
        self._cancel_requested = False
        if not keep_going:
            self.exit()
            return
        prompt = self.query_one("#prompt", PromptArea)
        prompt.disabled = False
        prompt.focus()

    # Interface `ChatScreen` : appelée depuis le thread de travail, exécutée dans le thread
    # de l'interface.

    def _on_ui_thread(self, function: Callable[..., Any], *args: Any) -> None:
        if threading.get_ident() == self._ui_thread:
            function(*args)
        else:
            self._call_ui(function, *args)

    def set_model(self, model: str) -> None:
        self._on_ui_thread(self._apply_model, model)

    def _apply_model(self, model: str) -> None:
        self._model_name = model
        self._draw_status()

    def set_context(self, context: tuple[int, int] | None) -> None:
        self._on_ui_thread(self._apply_context, context)

    def _apply_context(self, context: tuple[int, int] | None) -> None:
        self._context_usage = context
        self._draw_status()

    @property
    def context_style(self) -> str:
        """Couleur de l'occupation du contexte : vert, orange dès 60 %, rouge dès 80 %."""
        if self._context_usage is None:
            return ""
        level = context_level(*self._context_usage)
        return {"ok": "green", "warn": "yellow", "danger": "red"}[level]

    def _draw_status(self) -> None:
        status = Text(f" modèle {self._model_name} ")
        if self._context_usage is not None:
            used, limit = self._context_usage
            status.append("· contexte ")
            status.append(
                f"{format_tokens(used)} / {format_tokens(limit)} tokens "
                f"({context_percent(used, limit)} %)",
                style=self.context_style,
            )
        self.query_one("#status", Label).update(status)

    def start_progress(self) -> None:
        self._on_ui_thread(
            self._apply_progress, True, False, "L'agent traite votre demande"
        )

    def set_progress_message(self, message: str) -> None:
        self._on_ui_thread(self._apply_progress_message, message)

    def pause_progress(self) -> None:
        self._on_ui_thread(self._apply_pause, True)

    def resume_progress(self) -> None:
        self._on_ui_thread(self._apply_pause, False)

    def stop_progress(self) -> None:
        self._on_ui_thread(self._apply_progress, False, False, "")

    def _apply_progress(self, active: bool, paused: bool, message: str) -> None:
        if active:
            self._progress_started = perf_counter()
            self._progress_message = message
        self._progress_active = active
        self._progress_paused = paused
        self._draw_progress()

    def _apply_progress_message(self, message: str) -> None:
        self._progress_message = message

    def _apply_pause(self, paused: bool) -> None:
        self._progress_paused = paused
        self._draw_progress()

    def _draw_progress(self) -> None:
        label = self.query_one("#progress", Label)
        label.display = self._progress_active and not self._progress_paused
        if label.display:
            frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            elapsed = format_duration(perf_counter() - self._progress_started)
            # `Text` : le message peut contenir des crochets (nom d'outil du modèle) qu'un `str`
            # ferait lire comme du balisage.
            label.update(
                Text(
                    f"{frames[self._progress_frame % len(frames)]} "
                    f"{self._progress_message} ({elapsed})..."
                )
            )

    def _advance_spinner(self) -> None:
        self._progress_frame += 1
        if self._progress_active and not self._progress_paused:
            self._draw_progress()

    def _ask(self, question: str) -> bool:
        """Pose la question et attend la réponse ; appelé depuis le thread de travail."""
        answered = threading.Event()
        answer: list[bool] = []

        def finish(accepted: bool) -> None:
            answer.append(accepted)
            answered.set()

        if not self._call_ui(self._open_question, question, finish):
            return False
        answered.wait()
        return answer[0]

    def _open_question(self, question: str, finish: Callable[[bool], None]) -> None:
        """Affiche la question au-dessus de la saisie, qui est désactivée pendant le tour."""
        if self._cancel_requested:
            # Ctrl+C est arrivé avant l'ouverture de la question : refus, sans l'afficher.
            finish(False)
            return
        label = self.query_one("#question", Label)
        label.update(Text(safe(question)))
        label.display = True
        self._pending_answer = finish

    def _answer(self, accepted: bool) -> None:
        finish, self._pending_answer = self._pending_answer, None
        self.query_one("#question", Label).display = False
        if finish is not None:
            finish(accepted)

    def on_key(self, event: events.Key) -> None:
        """Pendant une question : `o` ou `y` accepte, `n`, Entrée ou Échap refuse (refus par
        défaut) ; les autres touches sont ignorées."""
        if self._pending_answer is None:
            return
        event.stop()
        event.prevent_default()
        key = event.key.lower()  # Verrouillage majuscules ou Maj : `O` vaut `o`
        if key in {"o", "y"}:
            self._answer(True)
        elif key in {"n", "enter", "escape"}:
            self._answer(False)

    def on_unmount(self) -> None:
        """Quelle que soit la façon de quitter (Ctrl+Q compris), une confirmation en attente
        est refusée (le thread du tour n'attend plus une réponse qui ne viendra pas) et le tour
        est annulé. Sans toucher à l'interface, déjà en cours de démontage."""
        finish, self._pending_answer = self._pending_answer, None
        if finish is not None:
            finish(False)
        # Le tour en cours ne doit pas continuer (outils, commandes) une fois l'interface fermée.
        self._session.cancel()

    def _refuse_confirmation(self) -> None:
        if self._pending_answer is not None:
            self._answer(False)

    def _apply_cancel(self, turn_id: int) -> None:
        """Transmet l'annulation à la session ; si le thread du tour n'est pas encore entré dans
        `handle` (Ctrl+C juste après Entrée), réessaie tant que ce même tour est en cours."""
        if self._busy and turn_id == self._turn_id and not self._session.cancel():
            self.set_timer(0.02, lambda: self._apply_cancel(turn_id))

    def action_interrupt(self) -> None:
        """Annule le tour en cours ; un second Ctrl+C, quand l'annulation tarde (appel du
        modèle non interruptible), ou un Ctrl+C sans tour en cours quitte."""
        self._refuse_confirmation()
        if self._busy and not self._cancel_requested:
            self._cancel_requested = True
            self._apply_cancel(self._turn_id)
        else:
            self.exit()

    def action_leave(self) -> None:
        self._refuse_confirmation()
        self._session.cancel()
        self.exit()
