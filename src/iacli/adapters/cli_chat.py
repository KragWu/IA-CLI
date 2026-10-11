"""Interactive console adapter for the agent graph."""

from collections.abc import Callable
import json
from pathlib import Path
import sys
from threading import Event, Lock, Thread
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

    def start_progress(self) -> None:
        self._progress_message = "L'agent traite votre demande"
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
            self._progress_stream.write(f"\r{frames[frame]} {message}...")
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
