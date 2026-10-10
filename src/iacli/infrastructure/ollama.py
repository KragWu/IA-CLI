"""HTTP gateway for Ollama and its model registry."""

import json
import platform
from typing import Callable, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen as _urlopen

from iacli.domain.events import DownloadProgress


class OllamaClient:
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
        raise ValueError(f"Aucun manifeste Ollama compatible avec {os_name}/{architecture}.")

    @staticmethod
    def _model_manifest_url(model: str, registry_url: str) -> str:
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
        del model
        return f"{self.base_url}/api/pull"

    def list_models(self) -> list[dict]:
        with self.urlopen(Request(self.list_models_url), timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("models", []), list):
            raise ValueError("La réponse Ollama doit contenir une liste de modèles.")
        return payload.get("models", [])

    def installed_models(self) -> set[str]:
        return {
            model["name"]
            for model in self.list_models()
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        }

    def is_model_installed(self, model: str) -> bool:
        return model in self.installed_models()

    def remote_model_size(self, model: str) -> int:
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
                f"{repository_url}/manifests/"
                f"{quote(descriptor['digest'], safe=':')}"
            )
            with self.urlopen(Request(manifest_url, headers=request.headers), timeout=10) as response:
                manifest = json.loads(response.read().decode("utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("Le registre Ollama a retourné un manifeste invalide.")

        descriptors = [manifest.get("config", {}), *manifest.get("layers", [])]
        if any(not isinstance(descriptor, Mapping) for descriptor in descriptors):
            raise ValueError("Le manifeste Ollama contient une couche invalide.")
        size = sum(int(descriptor.get("size", 0)) for descriptor in descriptors)
        if size <= 0:
            raise ValueError("Le manifeste Ollama ne fournit pas de taille exploitable.")
        return size

    def pull_model(
        self,
        model: str,
        progress: Callable[[DownloadProgress], None],
    ) -> None:
        request = Request(
            self.pull_url(model),
            data=json.dumps({"name": model, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.urlopen(request, timeout=3600) as response:
            self._read_pull_stream(response, progress)

    @staticmethod
    def _read_pull_stream(response, progress: Callable[[DownloadProgress], None]) -> None:
        succeeded = False
        while line := response.readline():
            if not line.strip():
                continue
            event = json.loads(line.decode("utf-8"))
            if not isinstance(event, dict):
                raise ValueError("Ollama a retourné un événement de téléchargement invalide.")
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
