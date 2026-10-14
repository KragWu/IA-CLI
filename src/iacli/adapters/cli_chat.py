"""Interactive console adapter for the agent graph."""

from collections.abc import Callable
import json
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import perf_counter
from typing import TextIO


class CliChatPresenter:
    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        progress_stream: TextIO = sys.stderr,
    ) -> None:
        self._input = input_fn
        self._output = output_fn
        self._progress_stream = progress_stream
        self._progress_stop = Event()
        self._progress_lock = Lock()
        self._progress_message = ""
        self._progress_started_at = 0.0
        self._progress_thread: Thread | None = None

    def start(self, workspace: Path, model: str) -> None:
        self._output(f"IA-CLI — modèle {model} — projet {workspace}")
        self._output("Entrez /exit ou /quit pour terminer.")

    def prompt(self) -> str:
        return self._input("vous> ")

    def show_response(self, response: str) -> None:
        self._output(f"\niacli> {response}\n")

    def show_tool_call(self, name: str, arguments: dict[str, object]) -> None:
        serialized_arguments = json.dumps(arguments, ensure_ascii=False)
        self._output(f"outil> {name}({serialized_arguments})")

    def show_tool_result(self, name: str) -> None:
        self._output(f"outil terminé> {name}")

    def show_action_timing(self, action: str, seconds: float) -> None:
        self._output(f"temps action> {action} : {self._format_duration(seconds)}")

    def show_total_duration(self, seconds: float) -> None:
        self._output(f"temps total> demande : {self._format_duration(seconds)}")

    def start_progress(self) -> None:
        self._progress_message = "L'agent traite votre demande"
        self._progress_started_at = perf_counter()
        self._output(f"{self._progress_message}...")
        self._resume_progress()

    def resume_progress(self) -> None:
        self._resume_progress()

    def _resume_progress(self) -> None:
        if not self._progress_stream.isatty():
            return
        if self._progress_thread is not None:
            return
        self._progress_stop.clear()
        self._progress_thread = Thread(target=self._animate_progress, daemon=True)
        self._progress_thread.start()

    def update_progress(self, message: str) -> None:
        with self._progress_lock:
            self._progress_message = message

    def stop_progress(self) -> None:
        if self._progress_thread is None:
            return
        self._progress_stop.set()
        self._progress_thread.join()
        self._progress_thread = None
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
            self._progress_stream.write(
                f"\r{frames[frame]} {message} ({duration})..."
            )
            self._progress_stream.flush()
            frame = (frame + 1) % len(frames)

    def confirm_file_write(self, path: str, diff: str) -> bool:
        self.stop_progress()
        self._output(f"Modification proposée pour {path} :")
        self._output(diff)
        try:
            return self._confirm("Appliquer cette modification ? [o/N] ")
        finally:
            self.resume_progress()

    def confirm_shell_command(self, command: str, workspace: Path) -> bool:
        self.stop_progress()
        self._output(f"Commande proposée dans {workspace} : {command}")
        try:
            return self._confirm("Exécuter cette commande ? [o/N] ")
        finally:
            self.resume_progress()

    def _confirm(self, prompt: str) -> bool:
        answer = self._input(prompt).strip().casefold()
        return answer in {"o", "oui", "y", "yes"}

    @staticmethod
    def _format_duration(seconds: float) -> str:
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
