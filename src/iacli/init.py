"""Initialisation de la configuration globale IA-CLI."""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen as _urlopen

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
MODEL_REQUIRED_BYTES = 5 * 1024**3

DEFAULT_CONFIG = '''[ollama]
base_url = "http://localhost:11434"
model = "qwen2.5-coder:7b"
'''

DEFAULT_INSTRUCTIONS = """# Instructions globales IA-CLI

- Répondre en français par défaut, sauf demande contraire.
- Privilégier des réponses précises, utiles et vérifiables.
- Respecter les conventions du projet en cours.
"""


def _available_disk_bytes(models_dir: Path) -> int:
    path = models_dir
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def _check_prerequisites(which: Callable[[str], str | None], output_fn: Callable[[str], None]) -> None:
    if sys.version_info < (3, 11):
        output_fn("Avertissement : Python 3.11 ou supérieur est requis.")

    for executable, label in (("git", "git"), ("rg", "ripgrep")):
        if which(executable) is None:
            output_fn(f"Avertissement : dépendance système manquante : {label}.")


def initialize(
    config_dir: Path | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    urlopen: Callable = _urlopen,
    which: Callable[[str], str | None] = shutil.which,
    disk_free_bytes: int | None = None,
) -> None:
    """Crée la configuration globale et propose le téléchargement du modèle."""
    target_dir = config_dir or Path.home() / ".config" / "iacli"
    _check_prerequisites(which, output_fn)

    target_dir.mkdir(parents=True, exist_ok=True)
    config_path = target_dir / "config.toml"
    instructions_path = target_dir / "IACLI.md"
    if not config_path.exists():
        config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    if not instructions_path.exists():
        instructions_path.write_text(DEFAULT_INSTRUCTIONS, encoding="utf-8")

    tags_url = f"{OLLAMA_URL}/api/tags"
    try:
        with urlopen(Request(tags_url), timeout=5) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        output_fn(f"Avertissement : Ollama n'est pas accessible sur {OLLAMA_URL} ({error}).")
        output_fn(f"Configuration créée dans {target_dir}.")
        output_fn("Initialisation réussie.")
        return

    installed_models = {model.get("name") for model in models}
    if DEFAULT_MODEL in installed_models:
        output_fn(f"Le modèle {DEFAULT_MODEL} est déjà installé.")
    else:
        models_dir = Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models")))
        free_bytes = disk_free_bytes if disk_free_bytes is not None else _available_disk_bytes(models_dir)
        if free_bytes < MODEL_REQUIRED_BYTES:
            output_fn(
                "Avertissement : espace disque insuffisant pour le modèle "
                f"{DEFAULT_MODEL} (environ 5 Gio requis)."
            )
        else:
            answer = input_fn(f"Télécharger le modèle {DEFAULT_MODEL} (environ 5 Gio) ? [O/n] ").strip().lower()
            if answer in ("", "o", "oui", "y", "yes"):
                pull_request = Request(
                    f"{OLLAMA_URL}/api/pull",
                    data=json.dumps({"name": DEFAULT_MODEL, "stream": False}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urlopen(pull_request, timeout=3600) as response:
                        result = json.loads(response.read().decode("utf-8"))
                    if result.get("status") != "success":
                        raise ValueError("Ollama n'a pas confirmé le téléchargement.")
                    output_fn(f"Modèle {DEFAULT_MODEL} téléchargé.")
                except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
                    output_fn(f"Avertissement : échec du téléchargement du modèle ({error}).")
            else:
                output_fn(f"Téléchargement du modèle {DEFAULT_MODEL} ignoré.")

    output_fn(f"Configuration créée dans {target_dir}.")
    output_fn("Initialisation réussie.")