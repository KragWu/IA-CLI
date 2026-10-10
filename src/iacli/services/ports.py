"""Ports required by application use cases."""

from pathlib import Path
from typing import Callable, Protocol

from iacli.domain.config import ConfigurationState, OllamaSettings
from iacli.domain.events import DownloadProgress, InitializationEvent


class ConfigurationGateway(Protocol):
    def ensure_configuration(self, directory: Path | None = None) -> ConfigurationState: ...


class SystemGateway(Protocol):
    def prerequisite_warnings(self) -> list[tuple[str, str]]: ...

    def available_model_disk_bytes(self) -> int: ...


class OllamaGateway(Protocol):
    def installed_models(self) -> set[str]: ...

    def remote_model_size(self, model: str) -> int: ...

    def pull_model(self, model: str, progress: Callable[[DownloadProgress], None]) -> None: ...


class OllamaGatewayFactory(Protocol):
    def create(self, settings: OllamaSettings) -> OllamaGateway: ...


class InitializationPresenter(Protocol):
    def present(self, event: InitializationEvent) -> None: ...

    def confirm_model_download(self, model: str, size_bytes: int) -> bool: ...
