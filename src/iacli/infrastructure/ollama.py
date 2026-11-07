"""Client HTTP (bibliothèque standard) pour Ollama et son registre de modèles."""

import json
import platform
from typing import Callable, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen as _urlopen

from iacli.domain.events import DownloadProgress
from iacli.domain.models import ModelInfo


class OllamaClient:
    """Implémente le port `OllamaGateway`.

    `urlopen` est injectable pour les tests. Un échec réseau lève `OSError`
    (`urllib.error.URLError`/`HTTPError`) ; une réponse inattendue lève `ValueError`.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        registry_url: str = "https://registry.ollama.ai",
        urlopen: Callable = _urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.registry_url = registry_url.rstrip("/")
        self.urlopen = urlopen

    @staticmethod
    def _platform_os_and_architecture() -> tuple[str, str]:
        """Système et architecture dans le vocabulaire du registre (ex. `windows`, `amd64`)."""
        machine = platform.machine().lower()
        architecture = {
            "amd64": "amd64",
            "x86_64": "amd64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }.get(machine, machine)
        system_name = platform.system().lower()
        os_name = (
            "darwin"
            if "darwin" in system_name or "mac" in system_name
            else "windows"
            if "win" in system_name
            else "linux"
        )
        return os_name, architecture

    @classmethod
    def _select_manifest_descriptor(cls, manifest: dict) -> dict:
        """Choisit, dans un index multi-plateforme, le manifeste de la machine locale."""
        candidates = manifest.get("manifests", [])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Aucun manifeste Ollama disponible.")
        if any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("Le manifeste Ollama contient un descripteur invalide.")

        os_name, architecture = cls._platform_os_and_architecture()
        for descriptor in candidates:
            descriptor_platform = descriptor.get("platform", {})
            if not isinstance(descriptor_platform, dict):
                continue
            if (
                descriptor_platform.get("architecture") == architecture
                and descriptor_platform.get("os") == os_name
            ):
                return descriptor
        if len(candidates) == 1 and not candidates[0].get("platform"):
            return candidates[0]
        raise ValueError(
            f"Aucun manifeste Ollama compatible avec {os_name}/{architecture}."
        )

    @staticmethod
    def _model_manifest_url(model: str, registry_url: str) -> str:
        """URL du manifeste : `qwen2.5-coder:7b` devient `<registre>/v2/library/qwen2.5-coder/manifests/7b`.

        Sans espace de noms le modèle est dans `library/` ; sans tag, c'est `latest`."""
        model_name, separator, tag = model.partition(":")
        if "/" not in model_name:
            model_name = f"library/{model_name}"
        tag = tag if separator else "latest"
        path = "/".join(quote(part, safe="") for part in model_name.split("/"))
        return f"{registry_url.rstrip('/')}/v2/{path}/manifests/{quote(tag, safe='')}"

    @property
    def list_models_url(self) -> str:
        return f"{self.base_url}/api/tags"

    def model_manifest_url(self, model: str) -> str:
        return self._model_manifest_url(model, self.registry_url)

    def pull_url(self, model: str) -> str:
        # `model` est ignoré : il est envoyé dans le corps de la requête, pas dans l'URL.
        del model
        return f"{self.base_url}/api/pull"

    def list_models(self) -> list[dict]:
        """Modèles installés tels que renvoyés par `GET /api/tags` (délai : 5 s)."""
        with self.urlopen(Request(self.list_models_url), timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(
            payload.get("models", []), list
        ):
            raise ValueError("La réponse Ollama doit contenir une liste de modèles.")
        return payload.get("models", [])

    def installed_models(self) -> set[str]:
        """Noms des modèles installés (champ `name`, tag inclus)."""
        return {
            model["name"]
            for model in self.list_models()
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        }

    def model_info(self, model: str) -> ModelInfo:
        """Capacités (ex. `tools`) et contexte maximal du modèle, via `POST /api/show`.

        Un champ absent de la réponse vaut None (« inconnu »)."""
        request = Request(
            f"{self.base_url}/api/show",
            data=json.dumps({"model": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("La réponse Ollama /api/show est invalide.")
        capabilities = payload.get("capabilities")
        return ModelInfo(
            capabilities=(
                frozenset(item for item in capabilities if isinstance(item, str))
                if isinstance(capabilities, list)
                else None
            ),
            context_length=self._context_length(payload),
        )

    @staticmethod
    def _context_length(payload: dict) -> int | None:
        """Cherche `<architecture>.context_length` dans `model_info`, à défaut n'importe quelle clé `*.context_length`."""
        model_info = payload.get("model_info")
        if not isinstance(model_info, dict):
            return None
        architecture = model_info.get("general.architecture")
        keys = (
            [f"{architecture}.context_length"] if isinstance(architecture, str) else []
        )
        keys += sorted(
            key
            for key in model_info
            if isinstance(key, str) and key.endswith(".context_length")
        )
        for key in keys:
            value = model_info.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    def is_model_installed(self, model: str) -> bool:
        """True si le nom exact (tag compris) figure parmi les modèles installés."""
        return model in self.installed_models()

    def remote_model_size(self, model: str) -> int:
        """Taille à télécharger : somme des tailles de la config et des couches du manifeste.

        Un index multi-plateforme déclenche une seconde requête vers le manifeste adapté
        à la machine. Un modèle inexistant lève `HTTPError` (code 404)."""
        request = Request(
            self.model_manifest_url(model),
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
        with self.urlopen(request, timeout=10) as response:
            manifest = json.loads(response.read().decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Le registre Ollama a retourné un manifeste invalide.")

        if "manifests" in manifest:
            descriptor = self._select_manifest_descriptor(manifest)
            repository_url = request.full_url.rsplit("/manifests/", 1)[0]
            manifest_url = (
                f"{repository_url}/manifests/{quote(descriptor['digest'], safe=':')}"
            )
            with self.urlopen(
                Request(manifest_url, headers=request.headers), timeout=10
            ) as response:
                manifest = json.loads(response.read().decode("utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("Le registre Ollama a retourné un manifeste invalide.")

        descriptors = [manifest.get("config", {}), *manifest.get("layers", [])]
        if any(not isinstance(descriptor, Mapping) for descriptor in descriptors):
            raise ValueError("Le manifeste Ollama contient une couche invalide.")
        size = sum(int(descriptor.get("size", 0)) for descriptor in descriptors)
        if size <= 0:
            raise ValueError(
                "Le manifeste Ollama ne fournit pas de taille exploitable."
            )
        return size

    def pull_model(
        self,
        model: str,
        progress: Callable[[DownloadProgress], None],
    ) -> None:
        """Télécharge le modèle via `POST /api/pull` en flux (délai maximal : 1 h)."""
        request = Request(
            self.pull_url(model),
            data=json.dumps({"name": model, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.urlopen(request, timeout=3600) as response:
            self._read_pull_stream(response, progress)

    @staticmethod
    def _read_pull_stream(
        response, progress: Callable[[DownloadProgress], None]
    ) -> None:
        """Lit le flux JSON ligne à ligne ; `ValueError` si Ollama signale une erreur
        ou si le flux se termine sans le statut final `success`."""
        succeeded = False
        while line := response.readline():
            if not line.strip():
                continue
            event = json.loads(line.decode("utf-8"))
            if not isinstance(event, dict):
                raise ValueError(
                    "Ollama a retourné un événement de téléchargement invalide."
                )
            if "error" in event:
                raise ValueError(event["error"])
            status = event.get("status", "Téléchargement en cours")
            completed = event.get("completed")
            total = event.get("total")
            progress(
                DownloadProgress(
                    status=status,
                    completed=completed if isinstance(completed, int) else None,
                    total=total if isinstance(total, int) else None,
                )
            )
            succeeded = succeeded or status == "success"
        if not succeeded:
            raise ValueError("Ollama a interrompu le téléchargement sans le terminer.")
