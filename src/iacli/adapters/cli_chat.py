"""Interactive console adapter for the agent graph."""

from collections.abc import Callable
from pathlib import Path


class CliChatPresenter:
    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._input = input_fn
        self._output = output_fn

    def start(self, workspace: Path, model: str) -> None:
        self._output(f"IA-CLI — modèle {model} — projet {workspace}")
        self._output("Entrez /exit ou /quit pour terminer.")

    def prompt(self) -> str:
        return self._input("vous> ")

    def show_response(self, response: str) -> None:
        self._output(f"\niacli> {response}\n")

    def confirm_file_write(self, path: str, diff: str) -> bool:
        self._output(f"Modification proposée pour {path} :")
        self._output(diff)
        return self._confirm("Appliquer cette modification ? [o/N] ")

    def confirm_shell_command(self, command: str, workspace: Path) -> bool:
        self._output(f"Commande proposée dans {workspace} : {command}")
        return self._confirm("Exécuter cette commande ? [o/N] ")

    def _confirm(self, prompt: str) -> bool:
        answer = self._input(prompt).strip().casefold()
        return answer in {"o", "oui", "y", "yes"}
