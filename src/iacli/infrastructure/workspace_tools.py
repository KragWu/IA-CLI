"""Outils de l'agent pour travailler dans le projet courant.

Lecture, liste et recherche sont libres ; l'écriture ou l'édition de fichier et la commande shell
exigent la confirmation de l'utilisateur. Seuls les chemins des outils de fichiers sont
confinés au projet : une commande shell, elle, n'est pas isolée.
"""

import contextlib
import difflib
import locale
import os
import signal
import subprocess
from collections.abc import Callable, Iterator
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, ParamSpec

from langchain_core.tools import BaseTool, tool

# Dossiers jamais parcourus par la liste, la recherche et la résolution d'un nom de fichier.
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
# Taille maximale (en octets) d'un fichier lu, parcouru ou écrit : évite de donner
# d'énormes fichiers au modèle.
MAX_FILE_BYTES = 1_000_000
# Taille maximale (en caractères) d'un résultat d'outil, sortie shell et diff compris,
# renvoyé dans la conversation.
MAX_TOOL_OUTPUT_CHARS = 30_000

# Durée maximale (en secondes) d'une commande shell avant son arrêt forcé.
MAX_TIME_FOR_SUBPROCESS = 600
# Délai (en secondes) pour récupérer la sortie une fois l'arbre de processus arrêté.
KILL_GRACE_SECONDS = 5
ToolParameters = ParamSpec("ToolParameters")


def _new_process_group_options() -> dict[str, Any]:
    """Options de `Popen` pour isoler la commande dans son propre groupe de processus,
    afin de pouvoir tuer toute sa descendance."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
    return {"start_new_session": True}


def _console_encoding() -> str:
    """Page de code des commandes shell : OEM sous Windows (cmd.exe), locale ailleurs."""
    if os.name == "nt":
        try:
            import ctypes

            return f"cp{ctypes.windll.kernel32.GetOEMCP()}"  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass
    return locale.getpreferredencoding(False)


def _decode_output(data: bytes) -> str:
    """UTF-8 d'abord (git, rg, python -X utf8...), sinon la page de code de la console."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(_console_encoding(), errors="replace")


# Ajouté à la fin d'un flux de sortie shell tronqué.
SHELL_TRUNCATION_MARKER = "\n[sortie tronquée]\n"


def _truncate(text: str, budget: int) -> str:
    """Garde le début de `text` pour tenir dans `budget` caractères, marqueur de troncature compris."""
    if len(text) <= budget:
        return text
    return (
        text[: max(budget - len(SHELL_TRUNCATION_MARKER), 0)] + SHELL_TRUNCATION_MARKER
    )


def _format_shell_result(header: str, stdout: str, stderr: str) -> str:
    """Assemble en-tête, stdout et stderr dans la limite des sorties d'outils.

    Chaque flux est tronqué séparément (début conservé, marqueur ajouté) : une sortie
    standard volumineuse ne doit pas faire disparaître l'erreur écrite sur stderr."""
    budget = MAX_TOOL_OUTPUT_CHARS - len(header)
    stdout_budget = max(budget // 2, budget - len(stderr))
    stdout_part = _truncate(stdout, stdout_budget)
    stderr_part = _truncate(stderr, budget - len(stdout_part))
    return header + stdout_part + stderr_part


# Signale, dans un diff, une dernière ligne sans retour à la ligne (comme git).
NO_FINAL_NEWLINE_MARKER = "\\ Pas de retour à la ligne à la fin du fichier\n"


def _unified_diff(previous: str, proposed: str, fromfile: str, tofile: str) -> str:
    """Diff unifié de deux textes en LF ; une dernière ligne sans `\\n` est signalée,
    sinon retirer ou ajouter ce retour à la ligne final serait invisible dans le diff."""
    lines = difflib.unified_diff(
        previous.splitlines(keepends=True),
        proposed.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
    )
    return "".join(
        line if line.endswith("\n") else f"{line}\n{NO_FINAL_NEWLINE_MARKER}"
        for line in lines
    )


def _count_occurrences(text: str, block: str) -> int:
    """Nombre d'occurrences de `block` dans `text`, chevauchements compris (`str.count`
    les ignore : « aa » compte une fois dans « aaa » alors qu'il y est à deux endroits)."""
    count = 0
    position = text.find(block)
    while position != -1:
        count += 1
        position = text.find(block, position + 1)
    return count


def _kill_process_tree(process: subprocess.Popen) -> None:
    """Arrête le shell et tous les processus qu'il a lancés, pas seulement le shell."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)  # type: ignore[attr-defined]


def create_workspace_tools(
    workspace: Path,
    confirm_file_write: Callable[[str, str], bool],
    confirm_shell_command: Callable[[str, Path], bool],
    on_action_timing: Callable[[str, float], None] | None = None,
) -> list[BaseTool]:
    """Crée les six outils (`read_file`, `list_files`, `search_files`, `write_file`,
    `edit_file`, `run_shell`).

    `confirm_file_write(chemin, diff)` et `confirm_shell_command(commande, dossier)`
    renvoient True si l'utilisateur accepte. `on_action_timing(outil, secondes)` est
    appelé après chaque outil. Une erreur levée par un outil est renvoyée au modèle.
    """
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
        """Chemin absolu de `path` (relatif au projet) ; `ValueError` s'il sort du projet (`..`, lien symbolique…)."""
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Le chemin doit rester dans le projet : {path}")
        return candidate

    def iter_files(directory: Path) -> Iterator[Path]:
        """Fichiers de `directory`, triés, hors dossiers ignorés et liens symboliques."""
        if any(
            part in IGNORED_DIRECTORIES for part in directory.relative_to(root).parts
        ):
            return
        for current, dirnames, filenames in os.walk(directory):
            # Élagage sur place : les arbres ignorés (.git, node_modules…) ne sont jamais parcourus.
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
        """Lire le contenu d'un fichier texte (UTF-8) du projet."""
        file_path = resolve_path(path)
        if file_path.is_dir():
            raise IsADirectoryError(
                f"Le chemin est un dossier : {path}. "
                "Utilisez list_files pour lister son contenu."
            )
        notice = ""
        if not file_path.is_file():
            candidates = [
                candidate
                for candidate in iter_files(root)
                if candidate.name.casefold() == Path(path).name.casefold()
            ]
            if len(candidates) == 1:
                file_path = candidates[0]
                notice = (
                    f"[Chemin demandé introuvable : {path} ; "
                    f"fichier lu : {file_path.relative_to(root).as_posix()}]\n"
                )
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
            return notice + content[:MAX_TOOL_OUTPUT_CHARS] + "\n[contenu tronqué]"
        return notice + content

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

    def check_content_size(path: str, content: str) -> None:
        """`ValueError` si `content` dépasse la taille maximale d'un fichier écrit."""
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > MAX_FILE_BYTES:
            raise ValueError(
                f"Contenu trop volumineux pour être écrit : {path} "
                f"({content_bytes} octets, maximum {MAX_FILE_BYTES})"
            )

    def read_existing_text(file_path: Path, path: str) -> str:
        """Contenu brut (fins de ligne conservées) d'un fichier existant du projet."""
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Fichier trop volumineux pour modification : {path}")
        try:
            with file_path.open(encoding="utf-8", newline="") as existing:
                return existing.read()
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Le fichier existant n'est pas encodé en UTF-8 : {path}"
            ) from error

    def detect_line_endings(raw_content: str) -> tuple[str, str]:
        """Fins de ligne à conserver (CRLF dès qu'il y en a une) et avertissement éventuel."""
        crlf_count = raw_content.count("\r\n")
        lf_count = raw_content.count("\n") - crlf_count
        if not crlf_count:
            return "\n", ""
        warning = ""
        if lf_count:
            warning = (
                "Attention : le fichier contient des fins de ligne mixtes "
                f"({crlf_count} CRLF, {lf_count} LF) ; toutes seront converties "
                "en CRLF.\n"
            )
        return "\r\n", warning

    def confirm_and_save(
        file_path: Path,
        previous: str,
        proposed: str,
        newline: str,
        warning: str,
        saved_message: str,
    ) -> str:
        """Affiche le diff `previous` -> `proposed` (textes en LF), puis écrit après accord."""
        relative_path = file_path.relative_to(root).as_posix()
        diff = _unified_diff(
            previous,
            proposed,
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
        if not diff and file_path.exists():
            return "Aucune modification : le fichier contient déjà ce texte."
        if not diff:
            diff = f"Création du fichier vide : {relative_path}\n"
        if len(diff) > MAX_TOOL_OUTPUT_CHARS:
            return (
                f"Modification refusée : la modification de {relative_path} est trop "
                f"volumineuse pour être affichée en entier ({len(diff)} caractères, "
                f"maximum {MAX_TOOL_OUTPUT_CHARS}). Découpez-la en plusieurs "
                "modifications plus petites."
            )
        if not confirm_file_write(relative_path, warning + diff):
            return f"Modification refusée par l'utilisateur : {relative_path}"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8", newline="") as target:
            target.write(proposed.replace("\n", newline))
        return f"{saved_message} : {relative_path}"

    @tool
    @measure_action("write_file")
    def write_file(path: str, content: str) -> str:
        """Créer un fichier ou remplacer tout son contenu, après confirmation utilisateur."""
        file_path = resolve_path(path)
        if file_path.exists() and not file_path.is_file():
            raise IsADirectoryError(f"Le chemin n'est pas un fichier : {path}")
        check_content_size(path, content)

        newline, warning = "\n", ""
        previous = ""
        if file_path.exists():
            existing_content = read_existing_text(file_path, path)
            newline, warning = detect_line_endings(existing_content)
            previous = existing_content.replace("\r\n", "\n")
        return confirm_and_save(
            file_path,
            previous,
            content.replace("\r\n", "\n"),
            newline,
            warning,
            "Fichier enregistré",
        )

    @tool
    @measure_action("edit_file")
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Remplacer un bloc de texte exact d'un fichier existant, après confirmation utilisateur.

        `old_string` doit apparaître exactement une fois dans le fichier : ajoute du
        contexte (lignes voisines) s'il est ambigu. Le reste du fichier n'est pas modifié."""
        if not old_string:
            raise ValueError("old_string ne peut pas être vide.")
        if old_string == new_string:
            raise ValueError("old_string et new_string sont identiques.")
        file_path = resolve_path(path)
        if file_path.is_dir():
            raise IsADirectoryError(f"Le chemin est un dossier : {path}")
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Fichier introuvable : {path}. "
                "Utilisez write_file pour créer un fichier."
            )

        raw_content = read_existing_text(file_path, path)
        newline, warning = detect_line_endings(raw_content)
        if warning:
            raise ValueError(
                f"Le fichier {path} contient des fins de ligne mixtes (CRLF et LF) : "
                "edit_file refuse de le modifier car il changerait aussi les lignes "
                "voisines. Utilisez write_file pour le réécrire en entier."
            )
        previous = raw_content.replace("\r\n", "\n")
        old_block = old_string.replace("\r\n", "\n")
        occurrences = _count_occurrences(previous, old_block)
        if occurrences == 0:
            raise ValueError(
                f"old_string introuvable dans {path} : il doit correspondre "
                "exactement au texte du fichier (espaces et retours à la ligne "
                "compris). Relisez le fichier avec read_file."
            )
        if occurrences > 1:
            raise ValueError(
                f"old_string apparaît {occurrences} fois dans {path} : ajoutez "
                "des lignes voisines pour l'identifier de façon unique."
            )
        proposed = previous.replace(old_block, new_string.replace("\r\n", "\n"), 1)
        check_content_size(path, proposed)
        return confirm_and_save(
            file_path, previous, proposed, newline, warning, "Fichier modifié"
        )

    @tool
    @measure_action("run_shell")
    def run_shell(command: str) -> str:
        """Exécuter une commande shell dans le projet après confirmation utilisateur.

        La réponse commence par le code de sortie. La commande n'est pas isolée du système."""
        if not command.strip():
            raise ValueError("La commande shell ne peut pas être vide.")
        if not confirm_shell_command(command, root):
            return "Exécution de la commande refusée par l'utilisateur."
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=root,
            # Ne pas partager la console du chat : une commande qui lit l'entrée
            # volerait ce que l'utilisateur tape.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_new_process_group_options(),
        )
        try:
            stdout, stderr = process.communicate(timeout=MAX_TIME_FOR_SUBPROCESS)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            try:
                stdout, stderr = process.communicate(timeout=KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                stdout, stderr = b"", b""
            return _format_shell_result(
                f"Commande interrompue après {MAX_TIME_FOR_SUBPROCESS} secondes.\n",
                _decode_output(stdout or b""),
                _decode_output(stderr or b""),
            )
        except KeyboardInterrupt:
            # Ctrl+C annule le tour : ne pas laisser la commande tourner en arrière-plan.
            _kill_process_tree(process)
            raise
        return _format_shell_result(
            f"Code de sortie : {process.returncode}\n",
            _decode_output(stdout),
            _decode_output(stderr),
        )

    return [
        read_file,
        list_files,
        search_files,
        write_file,
        edit_file,
        run_shell,
    ]
