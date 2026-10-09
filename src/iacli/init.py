"""Initialisation de la configuration globale IA-CLI."""

import os
import platform
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen as _urlopen

from iacli.ollama import OllamaClient

DEFAULT_CONFIG = '''[ollama]
base_url = "http://localhost:11434"
registry_url = "https://registry.ollama.ai"
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


def _platform_install_hint(executable: str) -> str:
    """Retourne une commande d'installation adaptée au système d'exploitation courant."""
    system_name = platform.system().lower()
    if "windows" in system_name:
        hints = {
            "git": "Installer Git via 'winget install --id Git.Git -e' ou depuis git-scm.com.",
            "rg": "Installer ripgrep via 'winget install --id BurntSushi.ripgrep.MSBuild -e'.",
        }
    elif "darwin" in system_name or "mac" in system_name:
        hints = {
            "git": "Installer Git avec 'brew install git'.",
            "rg": "Installer ripgrep avec 'brew install ripgrep'.",
        }
    else:
        hints = {
            "git": "Installer Git avec 'sudo apt install git' (Debian/Ubuntu), 'sudo dnf install git' (Fedora) ou 'sudo apk add git' (Alpine).",
            "rg": "Installer ripgrep avec 'sudo apt install ripgrep' (Debian/Ubuntu), 'sudo dnf install ripgrep' (Fedora) ou 'sudo apk add ripgrep' (Alpine).",
        }
    return hints.get(executable, f"Installez '{executable}' pour votre système d'exploitation.")


def _check_prerequisites(which: Callable[[str], str | None], output_fn: Callable[[str], None]) -> None:
    """Vérifie les dépendances minimales et donne une aide install rapide si besoin."""
    if sys.version_info < (3, 11):
        output_fn("Avertissement : Python 3.11 ou supérieur est requis.")

    for executable, label in (("git", "git"), ("rg", "ripgrep")):
        if which(executable) is None:
            output_fn(f"Avertissement : dépendance système manquante : {label}.")
            output_fn(f"Conseil : {_platform_install_hint(executable)}")


def _load_config(config_path: Path, output_fn: Callable[[str], None]) -> dict[str, str]:
    """Fusionne les valeurs par défaut avec les éventuelles overrides du fichier de config."""
    defaults: dict[str, str] = tomllib.loads(DEFAULT_CONFIG)["ollama"]
    loaded_config = defaults.copy()
    try:
        with config_path.open("rb") as config_file:
            # The global config file is the user's override layer; we keep the defaults
            # as the safety net when a value is missing or invalid.
            ollama_config = tomllib.load(config_file).get("ollama", {})
        if not isinstance(ollama_config, dict):
            raise ValueError("La section [ollama] doit être une table TOML.")
        for key in loaded_config:
            value = ollama_config.get(key, loaded_config[key])
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"La valeur ollama.{key} doit être une chaîne non vide.")
            loaded_config[key] = value.strip()
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        output_fn(f"Avertissement : configuration illisible ({error}), utilisation des valeurs par défaut.")
        return defaults
    return loaded_config


def _format_size(size_bytes: int) -> str:
    gib = size_bytes / 1024**3
    if gib >= 1:
        return f"{gib:.2f} Gio"
    return f"{size_bytes / 1024**2:.0f} Mio"


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
    # Use the standard config directory for the user, while keeping the code testable by
    # allowing a temporary folder to be injected via the `config_dir` parameter.
    target_dir = config_dir or Path.home() / ".config" / "iacli"
    output_fn("Étape 1/4 : vérification des prérequis système.")
    _check_prerequisites(which, output_fn)

    output_fn("Étape 2/4 : création de la configuration globale.")
    target_dir.mkdir(parents=True, exist_ok=True)
    config_path = target_dir / "config.toml"
    instructions_path = target_dir / "IACLI.md"
    if not config_path.exists():
        config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    if not instructions_path.exists():
        instructions_path.write_text(DEFAULT_INSTRUCTIONS, encoding="utf-8")

    config = _load_config(config_path, output_fn)
    ollama_url = config["base_url"].rstrip("/")
    registry_url = config["registry_url"].rstrip("/")
    default_model = config["model"]

    client = OllamaClient(base_url=ollama_url, registry_url=registry_url, urlopen=urlopen)
    output_fn(f"Étape 3/4 : vérification de la connexion à Ollama ({ollama_url}).")
    try:
        models = client.list_models()
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        output_fn(f"Avertissement : Ollama n'est pas accessible sur {ollama_url} ({error}).")
        output_fn("Étape 4/4 : préparation du modèle ignorée, Ollama est inaccessible.")
        output_fn(f"Configuration créée dans {target_dir}.")
        output_fn("Initialisation réussie.")
        return

    output_fn("Connexion à Ollama confirmée.")
    output_fn("Étape 4/4 : vérification du modèle et de l'espace disque.")
    # The model list returned by Ollama is the source of truth for local availability.
    installed_models = {model.get("name") for model in models}
    if default_model in installed_models:
        output_fn(f"Le modèle {default_model} est déjà installé, aucun téléchargement nécessaire.")
    else:
        output_fn(f"Le modèle {default_model} n'est pas présent localement.")
        try:
            model_size = client.remote_model_size(default_model)
        except HTTPError as error:
            if error.code == 404:
                output_fn(f"Avertissement : le modèle {default_model} n'existe pas dans le registre Ollama.")
            else:
                output_fn(f"Avertissement : impossible de vérifier le modèle dans le registre Ollama ({error}).")
        except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as error:
            output_fn(f"Avertissement : impossible de vérifier le modèle dans le registre Ollama ({error}).")
        else:
            formatted_size = _format_size(model_size)
            output_fn(f"Modèle {default_model} trouvé dans le registre Ollama ({formatted_size}).")
            models_dir = Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models")))
            free_bytes = disk_free_bytes if disk_free_bytes is not None else _available_disk_bytes(models_dir)
            if free_bytes < model_size:
                output_fn(
                    "Avertissement : espace disque insuffisant pour le modèle "
                    f"{default_model} ({formatted_size} requis, {_format_size(free_bytes)} disponibles)."
                )
            else:
                answer = input_fn(f"Télécharger le modèle {default_model} ({formatted_size}) ? [O/n] ").strip().lower()
                if answer in ("", "o", "oui", "y", "yes"):
                    output_fn(f"Téléchargement de {default_model} lancé, progression reçue d'Ollama :")
                    try:
                        client.pull_model(default_model, output_fn)
                        output_fn(f"Modèle {default_model} téléchargé.")
                    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyboardInterrupt) as error:
                        output_fn(f"Avertissement : échec ou interruption du téléchargement ({error}).")
                else:
                    output_fn(f"Téléchargement du modèle {default_model} ignoré.")

    output_fn(f"Configuration créée dans {target_dir}.")
    output_fn("Initialisation réussie.")