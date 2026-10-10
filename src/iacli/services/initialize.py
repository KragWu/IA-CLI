"""Initialization application use case."""

from pathlib import Path

from iacli.domain.events import (
    DownloadProgress,
    InitializationEvent,
    InitializationEventCode,
)
from iacli.services.ports import (
    ConfigurationGateway,
    InitializationPresenter,
    OllamaGateway,
    OllamaGatewayFactory,
    SystemGateway,
)


class InitializeApplication:
    def __init__(
        self,
        configuration: ConfigurationGateway,
        system: SystemGateway,
        ollama_factory: OllamaGatewayFactory,
        presenter: InitializationPresenter,
    ) -> None:
        self._configuration = configuration
        self._system = system
        self._ollama_factory = ollama_factory
        self._presenter = presenter

    def execute(self, config_directory: Path | None = None) -> None:
        self._emit(InitializationEventCode.STEP, number=1)
        for dependency, hint in self._system.prerequisite_warnings():
            self._emit(
                InitializationEventCode.PREREQUISITE_MISSING,
                dependency=dependency,
                hint=hint,
            )
            return

        self._emit(InitializationEventCode.STEP, number=2)
        configuration = self._configuration.ensure_configuration(config_directory)
        for warning in configuration.warnings:
            self._emit(InitializationEventCode.CONFIGURATION_WARNING, error=warning)
        ollama = self._ollama_factory.create(configuration.ollama)

        self._emit(
            InitializationEventCode.STEP,
            number=3,
            url=configuration.ollama.base_url,
        )
        try:
            installed_models = ollama.installed_models()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._emit(
                InitializationEventCode.OLLAMA_UNREACHABLE,
                url=configuration.ollama.base_url,
                error=str(error),
            )
        else:
            self._check_model(
                ollama,
                configuration.ollama.default_model,
                installed_models,
            )

        self._emit(
            InitializationEventCode.CONFIGURATION_CREATED,
            directory=str(configuration.directory),
        )
        self._emit(InitializationEventCode.INITIALIZATION_COMPLETED)

    def _check_model(
        self,
        ollama: OllamaGateway,
        model: str,
        installed_models: set[str],
    ) -> None:
        self._emit(InitializationEventCode.OLLAMA_CONNECTED)
        self._emit(InitializationEventCode.STEP, number=4)
        if model in installed_models:
            self._emit(InitializationEventCode.MODEL_INSTALLED, model=model)
            return

        self._emit(InitializationEventCode.MODEL_MISSING, model=model)
        try:
            model_size = ollama.remote_model_size(model)
        except OSError as error:
            if getattr(error, "code", None) == 404:
                self._emit(InitializationEventCode.MODEL_NOT_FOUND, model=model)
            else:
                self._emit(
                    InitializationEventCode.MODEL_LOOKUP_FAILED,
                    error=str(error),
                )
            return
        except (ValueError, KeyError, TypeError) as error:
            self._emit(
                InitializationEventCode.MODEL_LOOKUP_FAILED,
                error=str(error),
            )
            return

        self._emit(
            InitializationEventCode.MODEL_FOUND,
            model=model,
            size_bytes=model_size,
        )
        available_bytes = self._system.available_model_disk_bytes()
        if available_bytes < model_size:
            self._emit(
                InitializationEventCode.DISK_SPACE_INSUFFICIENT,
                model=model,
                required_bytes=model_size,
                available_bytes=available_bytes,
            )
            return

        if not self._presenter.confirm_model_download(model, model_size):
            self._emit(InitializationEventCode.PULL_SKIPPED, model=model)
            return

        self._emit(InitializationEventCode.PULL_STARTED, model=model)
        try:
            ollama.pull_model(
                model,
                lambda progress: self._present_progress(model, progress),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._emit(
                InitializationEventCode.PULL_FAILED,
                error=str(error),
            )
            return
        self._emit(InitializationEventCode.PULL_COMPLETED, model=model)

    def _present_progress(self, model: str, progress: DownloadProgress) -> None:
        values: dict[str, str | int] = {"model": model, "status": progress.status}
        if progress.completed is not None:
            values["completed"] = progress.completed
        if progress.total is not None:
            values["total"] = progress.total
        self._emit(InitializationEventCode.PULL_PROGRESS, **values)

    def _emit(
        self,
        code: InitializationEventCode,
        **values: str | int,
    ) -> None:
        self._presenter.present(InitializationEvent(code, values))
