"""User-approved tools for interacting with the current project workspace."""

import difflib
from functools import wraps
import os
import signal
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from time import perf_counter
from typing import ParamSpec

from langchain_core.tools import BaseTool, tool

IGNORED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
# Bound per-file reads and search scans to avoid feeding huge files to the model.
MAX_FILE_BYTES = 1_000_000
# Bound tool results (including shell output and diffs) sent back into the conversation.
MAX_TOOL_OUTPUT_CHARS = 30_000

MAX_TIME_FOR_SUBPROCESS = 600
# Time allowed to collect output once a timed-out process tree has been killed.
KILL_GRACE_SECONDS = 5
ToolParameters = ParamSpec("ToolParameters")


def _new_process_group_options() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    """Kill a shell and every process it started, not only the shell itself."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def create_workspace_tools(
    workspace: Path,
    confirm_file_write: Callable[[str, str], bool],
    confirm_shell_command: Callable[[str, Path], bool],
    on_action_timing: Callable[[str, float], None] | None = None,
) -> list[BaseTool]:
    root = workspace.resolve()

    def measure_action(
        name: str,
    ) -> Callable[
        [Callable[ToolParameters, str]],
        Callable[ToolParameters, str],
    ]:
        def decorator(
            function: Callable[ToolParameters, str],
        ) -> Callable[ToolParameters, str]:
            @wraps(function)
            def measured(
                *args: ToolParameters.args,
                **kwargs: ToolParameters.kwargs,
            ) -> str:
                started = perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    if on_action_timing is not None:
                        on_action_timing(name, perf_counter() - started)

            return measured

        return decorator

    def resolve_path(path: str) -> Path:
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Le chemin doit rester dans le projet : {path}")
        return candidate

    def iter_files(directory: Path) -> Iterator[Path]:
        if any(part in IGNORED_DIRECTORIES for part in directory.relative_to(root).parts):
            return
        for current, dirnames, filenames in os.walk(directory):
            # Prune in place so ignored trees (.git, node_modules...) are never read.
            dirnames[:] = sorted(
                name for name in dirnames if name not in IGNORED_DIRECTORIES
            )
            for name in sorted(filenames):
                path = Path(current) / name
                if not path.is_symlink() and path.is_file():
                    yield path

    @tool
    @measure_action("read_file")
    def read_file(path: str) -> str:
        """Lire le contenu d'un fichier du projet."""
        file_path = resolve_path(path)
        if not file_path.is_file():
            candidates = [
                candidate
                for candidate in iter_files(root)
                if candidate.name.casefold() == Path(path).name.casefold()
            ]
            if len(candidates) == 1:
                file_path = candidates[0]
            elif len(candidates) > 1:
                choices = ", ".join(
                    candidate.relative_to(root).as_posix()
                    for candidate in candidates[:10]
                )
                raise ValueError(
                    f"Le nom '{Path(path).name}' est ambigu. "
                    f"Précisez un chemin relatif au projet : {choices}"
                )
            else:
                raise FileNotFoundError(f"Fichier introuvable : {path}")
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Fichier trop volumineux pour lecture : {path}")
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Le fichier n'est pas un texte encodé en UTF-8 : {path}"
            ) from error
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            return content[:MAX_TOOL_OUTPUT_CHARS] + "\n[contenu tronqué]"
        return content

    @tool
    @measure_action("list_files")
    def list_files(path: str = ".") -> str:
        """Lister les fichiers du projet ou d'un sous-dossier."""
        target = resolve_path(path)
        if not target.is_dir():
            raise NotADirectoryError(f"Dossier introuvable : {path}")
        files = iter_files(target)
        results = []
        for file_path in files:
            results.append(file_path.relative_to(root).as_posix())
            if len(results) >= 200:
                results.append("[liste limitée à 200 fichiers]")
                break
        return "\n".join(results) if results else "Aucun fichier trouvé."

    @tool
    @measure_action("search_files")
    def search_files(query: str, path: str = ".") -> str:
        """Rechercher un texte dans les fichiers du projet, sans distinction de casse."""
        if not query:
            raise ValueError("Le texte recherché ne peut pas être vide.")
        target = resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Chemin introuvable : {path}")
        files = [target] if target.is_file() else iter_files(target)
        needle = query.casefold()
        matches = []
        skipped_large = 0
        skipped_binary = 0
        truncated = False
        for file_path in files:
            if file_path.is_symlink():
                continue
            if file_path.stat().st_size > MAX_FILE_BYTES:
                skipped_large += 1
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                skipped_binary += 1
                continue
            for line_number, line in enumerate(lines, 1):
                if needle in line.casefold():
                    matches.append(
                        f"{file_path.relative_to(root).as_posix()}:{line_number}: {line}"
                    )
                    if len(matches) >= 100:
                        truncated = True
                        break
            if truncated:
                matches.append("[résultats limités à 100 correspondances]")
                break
        if skipped_large or skipped_binary:
            matches.append(
                f"[fichiers ignorés : {skipped_large} trop volumineux, "
                f"{skipped_binary} non UTF-8]"
            )
        return "\n".join(matches) if matches else "Aucune correspondance."

    @tool
    @measure_action("write_file")
    def write_file(path: str, content: str) -> str:
        """Créer ou remplacer un fichier du projet après confirmation utilisateur."""
        file_path = resolve_path(path)
        if file_path.exists() and not file_path.is_file():
            raise IsADirectoryError(f"Le chemin n'est pas un fichier : {path}")
        if file_path.exists() and file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Fichier trop volumineux pour modification : {path}")

        newline = "\n"
        previous: list[str] = []
        if file_path.exists():
            try:
                with file_path.open(encoding="utf-8", newline="") as existing:
                    existing_content = existing.read()
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"Le fichier existant n'est pas encodé en UTF-8 : {path}"
                ) from error
            if "\r\n" in existing_content:
                newline = "\r\n"
            previous = existing_content.replace("\r\n", "\n").splitlines(keepends=True)
        content = content.replace("\r\n", "\n")
        proposed = content.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                previous,
                proposed,
                fromfile=f"a/{file_path.relative_to(root).as_posix()}",
                tofile=f"b/{file_path.relative_to(root).as_posix()}",
            )
        )
        if not diff:
            return "Aucune modification : le fichier contient déjà ce texte."
        relative_path = file_path.relative_to(root).as_posix()
        if len(diff) > MAX_TOOL_OUTPUT_CHARS:
            return (
                f"Modification refusée : la modification de {relative_path} est trop "
                f"volumineuse pour être affichée en entier ({len(diff)} caractères, "
                f"maximum {MAX_TOOL_OUTPUT_CHARS}). Découpez-la en plusieurs "
                "modifications plus petites."
            )
        if not confirm_file_write(relative_path, diff):
            return f"Modification refusée par l'utilisateur : {relative_path}"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8", newline="") as target:
            target.write(content.replace("\n", newline))
        return f"Fichier enregistré : {relative_path}"

    @tool
    @measure_action("run_shell")
    def run_shell(command: str) -> str:
        """Exécuter une commande shell dans le projet après confirmation utilisateur."""
        if not command.strip():
            raise ValueError("La commande shell ne peut pas être vide.")
        if not confirm_shell_command(command, root):
            return "Exécution de la commande refusée par l'utilisateur."
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_new_process_group_options(),
        )
        try:
            stdout, stderr = process.communicate(timeout=MAX_TIME_FOR_SUBPROCESS)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            try:
                stdout, stderr = process.communicate(timeout=KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            return (
                f"Commande interrompue après {MAX_TIME_FOR_SUBPROCESS} secondes.\n"
                f"{stdout or ''}{stderr or ''}"
            )[:MAX_TOOL_OUTPUT_CHARS]
        except KeyboardInterrupt:
            # Ctrl+C annule le tour : ne pas laisser la commande tourner en arrière-plan.
            _kill_process_tree(process)
            raise
        output = (
            f"Code de sortie : {process.returncode}\n"
            f"{stdout}{stderr}"
        )
        if len(output) > MAX_TOOL_OUTPUT_CHARS:
            output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n[sortie tronquée]"
        return output

    return [read_file, list_files, search_files, write_file, run_shell]
