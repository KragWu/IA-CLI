"""User-approved tools for interacting with the current project workspace."""

import difflib
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

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
MAX_FILE_BYTES = 1_000_000
MAX_TOOL_OUTPUT_CHARS = 30_000


def create_workspace_tools(
    workspace: Path,
    confirm_file_write: Callable[[str, str], bool],
    confirm_shell_command: Callable[[str, Path], bool],
) -> list[BaseTool]:
    root = workspace.resolve()

    def resolve_path(path: str) -> Path:
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Le chemin doit rester dans le projet : {path}")
        return candidate

    def iter_files(directory: Path) -> Iterator[Path]:
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if any(part in IGNORED_DIRECTORIES for part in relative.parts):
                continue
            if path.is_symlink():
                continue
            if path.is_file():
                yield path

    @tool
    def read_file(path: str) -> str:
        """Lire le contenu d'un fichier du projet."""
        file_path = resolve_path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Fichier trop volumineux pour lecture : {path}")
        content = file_path.read_text(encoding="utf-8")
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            return content[:MAX_TOOL_OUTPUT_CHARS] + "\n[contenu tronqué]"
        return content

    @tool
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
    def search_files(query: str, path: str = ".") -> str:
        """Rechercher un texte dans les fichiers du projet, sans distinction de casse."""
        if not query:
            raise ValueError("Le texte recherché ne peut pas être vide.")
        target = resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Chemin introuvable : {path}")
        files = [target] if target.is_file() else iter_files(target)
        matches = []
        skipped_large = 0
        skipped_binary = 0
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
                if query.casefold() in line.casefold():
                    matches.append(
                        f"{file_path.relative_to(root).as_posix()}:{line_number}: {line}"
                    )
                    if len(matches) >= 100:
                        matches.append("[résultats limités à 100 correspondances]")
                        return "\n".join(matches)
        if skipped_large or skipped_binary:
            matches.append(
                f"[fichiers ignorés : {skipped_large} trop volumineux, "
                f"{skipped_binary} non UTF-8]"
            )
        return "\n".join(matches) if matches else "Aucune correspondance."

    @tool
    def write_file(path: str, content: str) -> str:
        """Créer ou remplacer un fichier du projet après confirmation utilisateur."""
        file_path = resolve_path(path)
        if file_path.exists() and not file_path.is_file():
            raise IsADirectoryError(f"Le chemin n'est pas un fichier : {path}")
        if file_path.exists() and file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Fichier trop volumineux pour modification : {path}")

        previous = (
            file_path.read_text(encoding="utf-8").splitlines(keepends=True)
            if file_path.exists()
            else []
        )
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
        if len(diff) > MAX_TOOL_OUTPUT_CHARS:
            diff = diff[:MAX_TOOL_OUTPUT_CHARS] + "\n[aperçu du diff tronqué]"
        relative_path = file_path.relative_to(root).as_posix()
        if not confirm_file_write(relative_path, diff):
            return f"Modification refusée par l'utilisateur : {relative_path}"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Fichier enregistré : {relative_path}"

    @tool
    def run_shell(command: str) -> str:
        """Exécuter une commande shell dans le projet après confirmation utilisateur."""
        if not command.strip():
            raise ValueError("La commande shell ne peut pas être vide.")
        if not confirm_shell_command(command, root):
            return "Exécution de la commande refusée par l'utilisateur."
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            error_output = error.stderr or ""
            if isinstance(error_output, bytes):
                error_output = error_output.decode("utf-8", errors="replace")
            return (
                f"Commande interrompue après 120 secondes.\n{output}{error_output}"
            )[:MAX_TOOL_OUTPUT_CHARS]
        output = (
            f"Code de sortie : {result.returncode}\n"
            f"{result.stdout}{result.stderr}"
        )
        if len(output) > MAX_TOOL_OUTPUT_CHARS:
            output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n[sortie tronquée]"
        return output

    return [read_file, list_files, search_files, write_file, run_shell]
