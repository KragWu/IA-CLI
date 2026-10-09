"""Client léger pour interagir avec Ollama et son registre."""

import json
import platform
from typing import Callable
from urllib.parse import quote
from urllib.request import Request, urlopen as _urlopen


def _model_manifest_url(model: str, registry_url: str) -> str:
    """Construit l'URL du manifeste Docker/OCI pour un modèle donné."""
    model_name, separator, tag = model.partition(":")
    if "/" not in model_name:
        # Ollama registry uses a `library/` namespace for short names like `llama3`.
        model_name = f"library/{model_name}"
    tag = tag if separator else "latest"
    path = "/".join(quote(part, safe="") for part in model_name.split("/"))
    encoded_tag = quote(tag, safe="")
    return f"{registry_url.rstrip('/')}/v2/{path}/manifests/{encoded_tag}"


def _remote_model_size(model: str, registry_url: str, urlopen: Callable = _urlopen) -> int:
    request = Request(
        _model_manifest_url(model, registry_url),
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
        descriptor = OllamaClient._select_manifest_descriptor(manifest)
        repository_url = request.full_url.rsplit("/manifests/", 1)[0]
        manifest_url = f"{repository_url}/manifests/{quote(descriptor['digest'], safe=':')}"
        with urlopen(Request(manifest_url, headers=request.headers), timeout=10) as response:
            manifest = json.loads(response.read().decode("utf-8"))

    descriptors = [manifest.get("config", {}), *manifest.get("layers", [])]
    size = sum(int(descriptor.get("size", 0)) for descriptor in descriptors)
    if size <= 0:
        raise ValueError("Le manifeste Ollama ne fournit pas de taille exploitable.")
    return size


def _pull_model(response, output_fn: Callable[[str], None]) -> None:
    """Lit le flux SSE d'Ollama et affiche la progression du téléchargement."""
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


class OllamaClient:
    """Petit client public pour accéder à l'API Ollama et au registre."""

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
        """Normalise le système et l'architecture pour choisir le bon manifest registry."""
        machine = platform.machine().lower()
        architecture = {"amd64": "amd64", "x86_64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(
            machine, machine
        )
        system_name = platform.system().lower()
        os_name = "darwin" if "darwin" in system_name or "mac" in system_name else ("windows" if "win" in system_name else "linux")
        return os_name, architecture

    @classmethod
    def _select_manifest_descriptor(cls, manifest: dict) -> dict:
        candidates = manifest.get("manifests", [])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Aucun manifeste Ollama disponible.")

        os_name, architecture = cls._platform_os_and_architecture()
        compatible = [
            descriptor
            for descriptor in candidates
            if descriptor.get("platform", {}).get("architecture") == architecture
            and descriptor.get("platform", {}).get("os") in (os_name, "linux")
        ]
        if compatible:
            return compatible[0]
        if len(candidates) == 1:
            return candidates[0]
        return candidates[0]

    @property
    def list_models_url(self) -> str:
        return f"{self.base_url}/api/tags"

    def model_manifest_url(self, model: str) -> str:
        return _model_manifest_url(model, self.registry_url)

    def remote_model_size(self, model: str) -> int:
        return _remote_model_size(model, self.registry_url, self.urlopen)

    def pull_url(self, model: str) -> str:
        del model
        return f"{self.base_url}/api/pull"

    def list_models(self) -> list[dict]:
        """Retourne la liste des modèles installés côté Ollama."""
        with self.urlopen(Request(self.list_models_url), timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("models", [])

    def installed_models(self) -> set[str]:
        """Extrait les noms de modèles installés sous forme de set pour un test rapide."""
        return {model.get("name") for model in self.list_models() if isinstance(model, dict) and model.get("name")}

    def is_model_installed(self, model: str) -> bool:
        """Vérifie si un modèle donné est déjà présent localement."""
        return model in self.installed_models()

    def pull_model(self, model: str, output_fn: Callable[[str], None]) -> None:
        """Télécharge un modèle en flux, tout en relayant la progression à l'utilisateur."""
        request = Request(
            self.pull_url(model),
            data=json.dumps({"name": model, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.urlopen(request, timeout=3600) as response:
            _pull_model(response, output_fn)
