"""Ports required by application use cases."""

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Protocol

from langchain_core.tools import BaseTool

from iacli.domain.config import ConfigurationState, OllamaSettings
from iacli.domain.events import DownloadProgress, InitializationEvent


class ConfigurationGateway(Protocol):
    def ensure_configuration(self, directory: Path | None = None) -> ConfigurationState: ...


class SystemGateway(Protocol):
    def prerequisite_warnings(self) -> list[tuple[str, str]]: ...

    def available_model_disk_bytes(self) -> int: ...


class OllamaGateway(Protocol):
    def installed_models(self) -> set[str]: ...

    def model_capabilities(self, model: str) -> set[str] | None:
        """Capacités annoncées par Ollama, ou None si le serveur ne les expose pas."""
        ...

    def remote_model_size(self, model: str) -> int: ...

    def pull_model(self, model: str, progress: Callable[[DownloadProgress], None]) -> None: ...


class OllamaGatewayFactory(Protocol):
    def create(self, settings: OllamaSettings) -> OllamaGateway: ...


class InitializationPresenter(Protocol):
    def present(self, event: InitializationEvent) -> None: ...

    def confirm_model_download(self, model: str, size_bytes: int) -> bool: ...


class ChatPresenter(Protocol):
    def start(self, workspace: Path, model: str) -> None: ...

    def prompt(self) -> str: ...

    def show_response(self, response: str) -> None: ...

    def show_error(self, message: str) -> None: ...

    def show_tool_call(self, name: str, arguments: dict[str, object]) -> None: ...

    def show_tool_result(self, name: str) -> None: ...

    def show_action_timing(self, action: str, seconds: float) -> None: ...

    def show_total_duration(self, seconds: float) -> None: ...

    def start_progress(self) -> None: ...

    def update_progress(self, message: str) -> None: ...

    def stop_progress(self) -> None: ...

    def confirm_file_write(self, path: str, diff: str) -> bool: ...

    def confirm_shell_command(self, command: str, workspace: Path) -> bool: ...


class ToolsFactory(Protocol):
    def __call__(
        self,
        workspace: Path,
        confirm_file_write: Callable[[str, str], bool],
        confirm_shell_command: Callable[[str, Path], bool],
        on_action_timing: Callable[[str, float], None] | None = None,
    ) -> Sequence[BaseTool]: ...
