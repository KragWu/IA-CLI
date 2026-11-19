"""Adapter console du chat : invite, affichage, confirmations et indicateur animé."""

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from time import perf_counter
from typing import TextIO

from iacli.domain.commands import SlashCommand
from iacli.domain.status import PromptStatus, format_tokens
from iacli.domain.text import neutralize_control_characters as safe

# Préfixe des réponses de l'agent ; le rendu markdown en tient compte pour sa largeur.
RESPONSE_PREFIX = "iacli> "
# Colonnes réservées au rendu markdown : le préfixe, plus une colonne libre (écrire dans la
# dernière colonne provoque, dans certains terminaux, un retour à la ligne en trop).
RESPONSE_MARGIN = len(RESPONSE_PREFIX) + 1


def _render_or_plain(render: Callable[[str], str], text: str) -> str:
    """Texte rendu ; en cas d'échec du rendu, le texte brut : une réponse n'est jamais perdue."""
    try:
        return render(text)
    except Exception:
        return text


class CliChatPresenter:
    """Implémente le port `ChatPresenter` pour un terminal.

    Les lignes de résultat vont sur la sortie `output_fn` (stdout) ; l'indicateur
    animé s'écrit sur `progress_stream` (stderr) et seulement si c'est un terminal
    interactif. `input_fn`, `output_fn`, `render_markdown` et `render_diff` (rendu des
    réponses et des diffs, brut si absent) sont injectables pour les tests.
    """

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        progress_stream: TextIO = sys.stderr,
        render_markdown: Callable[[str], str] | None = None,
        render_diff: Callable[[str], str] | None = None,
        prompt_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._prompt_fn = prompt_fn or input_fn
        self._model = ""
        self._context: tuple[int, int] | None = None
        self._render_diff = render_diff
        self._render_markdown = render_markdown
        self._input = input_fn
        self._output = output_fn
        self._progress_stream = progress_stream
        self._progress_stop = Event()
        self._progress_lock = Lock()
        self._output_lock = RLock()
        # Sérialise démarrage et arrêt du thread d'animation : les confirmations
        # d'outils les appellent depuis le thread d'exécution des outils.
        self._thread_lock = RLock()
        self._progress_message = ""
        self._progress_started_at = 0.0
        self._progress_thread: Thread | None = None

    def _emit(self, line: str) -> None:
        """Affiche une ligne en effaçant d'abord l'indicateur animé pour ne pas le mélanger au texte."""
        with self._output_lock:
            if self._progress_thread is not None:
                self._progress_stream.write("\r\033[2K")
                self._progress_stream.flush()
            self._output(line)

    def start(self, workspace: Path, model: str) -> None:
        self._model = model
        self._emit(f"IA-CLI — modèle {model} — projet {workspace}")
        self._emit("Entrez /help pour les commandes, /exit ou /quit pour terminer.")

    def prompt(self) -> str:
        return self._prompt_fn("vous> ")

    def prompt_status(self) -> PromptStatus:
        used, limit = self._context or (None, None)
        return PromptStatus(self._model, used, limit)

    def show_response(self, response: str) -> None:
        response = safe(response)
        if self._render_markdown is not None:
            # Les lignes suivantes s'alignent sous le texte, après le préfixe.
            indent = " " * len(RESPONSE_PREFIX)
            rendered = _render_or_plain(self._render_markdown, response)
            response = rendered.replace("\n", "\n" + indent)
        self._emit(f"\n{RESPONSE_PREFIX}{response}\n")

    def show_error(self, message: str) -> None:
        self._emit(f"erreur> {safe(message)}")

    def show_notice(self, message: str) -> None:
        self._emit(message)

    def show_help(self, commands: Sequence[SlashCommand]) -> None:
        self._emit("Commandes disponibles :")
        for command in commands:
            self._emit(f"  {command.name:<10} {command.description}")

    def show_cancelled(self) -> None:
        self._emit("Génération annulée")

    def show_tool_call(self, name: str, arguments: dict[str, object]) -> None:
        serialized_arguments = json.dumps(arguments, ensure_ascii=False)
        self._emit(f"outil> {safe(name)}({safe(serialized_arguments)})")

    def show_tool_result(self, name: str) -> None:
        self._emit(f"outil terminé> {name}")

    def show_action_timing(self, action: str, seconds: float) -> None:
        self._emit(f"temps action> {action} : {self._format_duration(seconds)}")

    def show_total_duration(self, seconds: float) -> None:
        self._emit(f"temps total> demande : {self._format_duration(seconds)}")

    def show_context_usage(self, used: int, limit: int) -> None:
        self._context = (used, limit)
        self._emit(
            f"contexte> {format_tokens(used)} / {format_tokens(limit)} "
            f"tokens ({round(100 * used / limit)} %)"
        )

    def clear_context_usage(self) -> None:
        self._context = None

    def show_context_warning(self, used: int, limit: int) -> None:
        self._emit(
            f"attention> contexte utilisé à {round(100 * used / limit)} % : au-delà de "
            f"{format_tokens(limit)} tokens, Ollama peut tronquer les plus "
            "anciens messages de la conversation."
        )

    def start_progress(self) -> None:
        """Démarre le chronomètre du tour et l'animation (terminal interactif uniquement)."""
        self._progress_message = "L'agent traite votre demande"
        self._progress_started_at = perf_counter()
        self._emit(f"{self._progress_message}...")
        self._resume_progress()

    def resume_progress(self) -> None:
        """Relance l'animation après une confirmation (le chronomètre n'est pas remis à zéro)."""
        self._resume_progress()

    def _resume_progress(self) -> None:
        if not self._progress_stream.isatty():
            return
        with self._thread_lock:
            if self._progress_thread is not None:
                return
            self._progress_stop.clear()
            thread = Thread(target=self._animate_progress, daemon=True)
            thread.start()
            self._progress_thread = thread

    def update_progress(self, message: str) -> None:
        """Change le message affiché à côté de l'animation."""
        with self._progress_lock:
            self._progress_message = safe(message)

    def stop_progress(self) -> None:
        """Arrête l'animation et efface sa ligne ; sans effet si elle n'est pas lancée."""
        with self._thread_lock:
            thread = self._progress_thread
            if thread is None:
                return
            self._progress_stop.set()
            thread.join()
            self._progress_thread = None
            with self._output_lock:
                self._progress_stream.write("\r\033[2K")
                self._progress_stream.flush()

    def _animate_progress(self) -> None:
        frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
        frame = 0
        while not self._progress_stop.wait(0.1):
            with self._progress_lock:
                message = self._progress_message
            elapsed = perf_counter() - self._progress_started_at
            duration = self._format_duration(elapsed)
            with self._output_lock:
                if self._progress_stop.is_set():
                    break
                self._progress_stream.write(
                    f"\r{frames[frame]} {message} ({duration})..."
                )
                self._progress_stream.flush()
            frame = (frame + 1) % len(frames)

    def confirm_file_write(self, path: str, diff: str) -> bool:
        """Suspend l'animation, montre le diff et demande confirmation (refus par défaut)."""
        self.stop_progress()
        diff = safe(diff)
        self._emit(f"Modification proposée pour {safe(path)} :")
        self._emit(
            _render_or_plain(self._render_diff, diff) if self._render_diff else diff
        )
        try:
            return self._confirm("Appliquer cette modification ? [o/N] ")
        finally:
            self.resume_progress()

    def confirm_shell_command(self, command: str, workspace: Path) -> bool:
        """Suspend l'animation, montre la commande et demande confirmation (refus par défaut)."""
        self.stop_progress()
        self._emit(f"Commande proposée dans {safe(str(workspace))} : {safe(command)}")
        try:
            return self._confirm("Exécuter cette commande ? [o/N] ")
        finally:
            self.resume_progress()

    def _confirm(self, prompt: str) -> bool:
        """True pour `o`, `oui`, `y` ou `yes` ; toute autre réponse, vide comprise, est un refus."""
        answer = self._input(prompt).strip().casefold()
        return answer in {"o", "oui", "y", "yes"}

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Durée lisible : `850 ms`, `1.25 s`, `2 min 05 s` ou `1 h 02 min 03 s`."""
        if seconds < 1:
            milliseconds = seconds * 1000
            if milliseconds < 10:
                return f"{milliseconds:.1f} ms"
            return f"{milliseconds:.0f} ms"
        if seconds < 60:
            return f"{seconds:.2f} s"
        whole_seconds = round(seconds)
        if seconds < 3600:
            minutes, remaining_seconds = divmod(whole_seconds, 60)
            return f"{minutes} min {remaining_seconds:02d} s"
        hours, remainder = divmod(whole_seconds, 3600)
        minutes, remaining_seconds = divmod(remainder, 60)
        return f"{hours} h {minutes:02d} min {remaining_seconds:02d} s"
