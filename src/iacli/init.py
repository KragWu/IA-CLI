"""Initialisation de la configuration globale IA-CLI."""

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen as _urlopen

OLLAMA_URL = "http://localhost:11434"
OLLAMA_REGISTRY_URL = "https://registry.ollama.ai"
DEFAULT_MODEL = "qwen2.5-coder:7b"

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


def _pull_model(response, output_fn: Callable[[str], None]) -> None:
    succeeded = False
    while line := response.readline():
        if not line.strip():
            continue
        event = json.loads(line.decode("utf-8"))
        if "error" in event:
            raise ValueError(event["error"])

        status = event.get("status", "Téléchargement en cours")
        total = event.get("total")
        completed = event.get("completed")
        if isinstance(total, int) and total > 0 and isinstance(completed, int):
            percentage = min(100, completed * 100 // total)
            completed_mib = completed / 1024**2
            total_mib = total / 1024**2
            output_fn(f"{status} : {percentage}% ({completed_mib:.1f}/{total_mib:.1f} Mio)")
        else:
            output_fn(f"Téléchargement : {status}")

        if status == "success":
            succeeded = True

    if not succeeded:
        raise ValueError("Ollama a interrompu le téléchargement sans le terminer.")


def _model_manifest_url(model: str) -> str:
    model_name, separator, tag = model.partition(":")
    if "/" not in model_name:
        model_name = f"library/{model_name}"
    tag = tag if separator else "latest"
    path = "/".join(quote(part, safe="") for part in model_name.split("/"))
    encoded_tag = quote(tag, safe="")
    return f"{OLLAMA_REGISTRY_URL}/v2/{path}/manifests/{encoded_tag}"


def _remote_model_size(model: str, urlopen: Callable) -> int:
    request = Request(
        _model_manifest_url(model),
        headers={
            "Accept": ", ".join(
                (
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.v2+json",
                    "application/vnd.oci.image.index.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                )
            )
        },
    )
    with urlopen(request, timeout=10) as response:
        manifest = json.loads(response.read().decode("utf-8"))

    if "manifests" in manifest:
        machine = platform.machine().lower()
        architecture = {"amd64": "amd64", "x86_64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(
            machine, machine
        )
        candidates = [
            descriptor
            for descriptor in manifest["manifests"]
            if descriptor.get("platform", {}).get("architecture") == architecture
            and descriptor.get("platform", {}).get("os") == "linux"
        ]
        if not candidates and len(manifest["manifests"]) == 1:
            candidates = manifest["manifests"]
        if not candidates:
            raise ValueError(f"Aucun manifeste Ollama trouvé pour la plateforme {architecture}.")
        repository_url = request.full_url.rsplit("/manifests/", 1)[0]
        manifest_url = f"{repository_url}/manifests/{quote(candidates[0]['digest'], safe=':')}"
        with urlopen(Request(manifest_url, headers=request.headers), timeout=10) as response:
            manifest = json.loads(response.read().decode("utf-8"))

    descriptors = [manifest.get("config", {}), *manifest.get("layers", [])]
    size = sum(int(descriptor.get("size", 0)) for descriptor in descriptors)
    if size <= 0:
        raise ValueError("Le manifeste Ollama ne fournit pas de taille exploitable.")
    return size


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

    tags_url = f"{OLLAMA_URL}/api/tags"
    output_fn(f"Étape 3/4 : vérification de la connexion à Ollama ({OLLAMA_URL}).")
    try:
        with urlopen(Request(tags_url), timeout=5) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        output_fn(f"Avertissement : Ollama n'est pas accessible sur {OLLAMA_URL} ({error}).")
        output_fn("Étape 4/4 : préparation du modèle ignorée, Ollama est inaccessible.")
        output_fn(f"Configuration créée dans {target_dir}.")
        output_fn("Initialisation réussie.")
        return

    output_fn("Connexion à Ollama confirmée.")
    output_fn("Étape 4/4 : vérification du modèle et de l'espace disque.")
    installed_models = {model.get("name") for model in models}
    if DEFAULT_MODEL in installed_models:
        output_fn(f"Le modèle {DEFAULT_MODEL} est déjà installé, aucun téléchargement nécessaire.")
    else:
        output_fn(f"Le modèle {DEFAULT_MODEL} n'est pas présent localement.")
        try:
            model_size = _remote_model_size(DEFAULT_MODEL, urlopen)
        except HTTPError as error:
            if error.code == 404:
                output_fn(f"Avertissement : le modèle {DEFAULT_MODEL} n'existe pas dans le registre Ollama.")
            else:
                output_fn(f"Avertissement : impossible de vérifier le modèle dans le registre Ollama ({error}).")
        except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as error:
            output_fn(f"Avertissement : impossible de vérifier le modèle dans le registre Ollama ({error}).")
        else:
            formatted_size = _format_size(model_size)
            output_fn(f"Modèle {DEFAULT_MODEL} trouvé dans le registre Ollama ({formatted_size}).")
            models_dir = Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models")))
            free_bytes = disk_free_bytes if disk_free_bytes is not None else _available_disk_bytes(models_dir)
            if free_bytes < model_size:
                output_fn(
                    "Avertissement : espace disque insuffisant pour le modèle "
                    f"{DEFAULT_MODEL} ({formatted_size} requis, {_format_size(free_bytes)} disponibles)."
                )
            else:
                answer = input_fn(f"Télécharger le modèle {DEFAULT_MODEL} ({formatted_size}) ? [O/n] ").strip().lower()
                if answer in ("", "o", "oui", "y", "yes"):
                    output_fn(f"Téléchargement de {DEFAULT_MODEL} lancé, progression reçue d'Ollama :")
                    pull_request = Request(
                        f"{OLLAMA_URL}/api/pull",
                        data=json.dumps({"name": DEFAULT_MODEL, "stream": True}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    try:
                        with urlopen(pull_request, timeout=3600) as response:
                            _pull_model(response, output_fn)
                        output_fn(f"Modèle {DEFAULT_MODEL} téléchargé.")
                    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
                        output_fn(f"Avertissement : échec du téléchargement du modèle ({error}).")
                else:
                    output_fn(f"Téléchargement du modèle {DEFAULT_MODEL} ignoré.")

    output_fn(f"Configuration créée dans {target_dir}.")
    output_fn("Initialisation réussie.")