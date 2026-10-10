import json
import sys
from urllib.error import HTTPError, URLError

import pytest

from iacli.adapters.cli_presenter import CliPresenter
from iacli.domain.config import OllamaSettings
from iacli.domain.events import (
    DownloadProgress,
    InitializationEvent,
    InitializationEventCode,
)
from iacli.infrastructure.configuration import (
    DEFAULT_INSTRUCTIONS,
    FileConfigurationGateway,
)
from iacli.infrastructure.ollama import OllamaClient
from iacli.services.initialize import InitializeApplication


class FakeSystem:
    def __init__(self, free_bytes: int = 10 * 1024**3, warnings=None) -> None:
        self.free_bytes = free_bytes
        self.warnings = warnings or []

    def prerequisite_warnings(self) -> list[tuple[str, str]]:
        return self.warnings

    def available_model_disk_bytes(self) -> int:
        return self.free_bytes


class FakeOllama:
    def __init__(
        self,
        models: set[str] | None = None,
        model_size: int = 1024,
        list_error: BaseException | None = None,
        size_error: BaseException | None = None,
        pull_error: BaseException | None = None,
    ) -> None:
        self.models = models or set()
        self.model_size = model_size
        self.list_error = list_error
        self.size_error = size_error
        self.pull_error = pull_error
        self.size_checks = []
        self.pulls = []

    def installed_models(self) -> set[str]:
        if self.list_error:
            raise self.list_error
        return self.models

    def remote_model_size(self, model: str) -> int:
        self.size_checks.append(model)
        if self.size_error:
            raise self.size_error
        return self.model_size

    def pull_model(self, model: str, progress) -> None:
        self.pulls.append(model)
        if self.pull_error:
            raise self.pull_error
        progress(DownloadProgress("downloading", completed=50, total=100))


class FakeOllamaFactory:
    def __init__(self, client: FakeOllama) -> None:
        self.client = client
        self.settings = []

    def create(self, settings: OllamaSettings) -> FakeOllama:
        self.settings.append(settings)
        return self.client


class RecordingPresenter:
    def __init__(self, confirm: bool = False) -> None:
        self.confirm = confirm
        self.events: list[InitializationEvent] = []
        self.confirmations = []

    def present(self, event: InitializationEvent) -> None:
        self.events.append(event)

    def confirm_model_download(self, model: str, size_bytes: int) -> bool:
        self.confirmations.append((model, size_bytes))
        return self.confirm

    def codes(self) -> list[InitializationEventCode]:
        return [event.code for event in self.events]


def create_use_case(
    ollama: FakeOllama,
    *,
    presenter: RecordingPresenter | None = None,
    system: FakeSystem | None = None,
) -> tuple[InitializeApplication, RecordingPresenter, FakeOllamaFactory]:
    recording_presenter = presenter or RecordingPresenter()
    factory = FakeOllamaFactory(ollama)
    use_case = InitializeApplication(
        configuration=FileConfigurationGateway(),
        system=system or FakeSystem(),
        ollama_factory=factory,
        presenter=recording_presenter,
    )
    return use_case, recording_presenter, factory


def test_configuration_gateway_creates_files_and_preserves_user_content(tmp_path):
    config_directory = tmp_path / "iacli"
    config_directory.mkdir()
    config_path = config_directory / "config.toml"
    config_path.write_text(
        '[ollama]\nbase_url = "http://ollama.test:11434/"\n'
        'registry_url = "https://registry.test/"\nmodel = "custom:latest"\n',
        encoding="utf-8",
    )
    instructions_path = config_directory / "IACLI.md"
    instructions_path.write_text("User-owned instructions", encoding="utf-8")

    state = FileConfigurationGateway().ensure_configuration(config_directory)

    assert state.ollama == OllamaSettings(
        "http://ollama.test:11434",
        "https://registry.test",
        "custom:latest",
    )
    assert config_path.read_text(encoding="utf-8").count("custom:latest") == 1
    assert instructions_path.read_text(encoding="utf-8") == "User-owned instructions"


def test_configuration_gateway_creates_default_instructions(tmp_path):
    state = FileConfigurationGateway().ensure_configuration(tmp_path / "iacli")

    assert (state.directory / "config.toml").is_file()
    assert (state.directory / "IACLI.md").read_text(encoding="utf-8") == DEFAULT_INSTRUCTIONS
    assert state.ollama.default_model == "qwen2.5-coder:7b"


def test_configuration_gateway_fills_omitted_settings_with_defaults(tmp_path):
    config_directory = tmp_path / "iacli"
    config_directory.mkdir()
    (config_directory / "config.toml").write_text(
        '[ollama]\nbase_url = "http://ollama.test"\n',
        encoding="utf-8",
    )

    state = FileConfigurationGateway().ensure_configuration(config_directory)

    assert state.ollama == OllamaSettings(
        "http://ollama.test",
        "https://registry.ollama.ai",
        "qwen2.5-coder:7b",
    )


def test_use_case_skips_registry_and_prompt_when_model_is_installed(tmp_path):
    use_case, presenter, factory = create_use_case(
        FakeOllama(models={"qwen2.5-coder:7b"}),
    )

    use_case.execute(tmp_path / "iacli")

    assert presenter.codes().count(InitializationEventCode.MODEL_INSTALLED) == 1
    assert presenter.codes().count(InitializationEventCode.INITIALIZATION_COMPLETED) == 1
    assert presenter.confirmations == []
    assert factory.settings[0].default_model == "qwen2.5-coder:7b"


def test_use_case_pulls_model_after_confirmation_and_reports_progress(tmp_path):
    ollama = FakeOllama(model_size=2048)
    presenter = RecordingPresenter(confirm=True)
    use_case, presenter, _ = create_use_case(
        ollama,
        presenter=presenter,
    )

    use_case.execute(tmp_path / "iacli")

    assert ollama.size_checks == ["qwen2.5-coder:7b"]
    assert ollama.pulls == ["qwen2.5-coder:7b"]
    assert presenter.codes().count(InitializationEventCode.PULL_PROGRESS) == 1
    assert presenter.codes().count(InitializationEventCode.PULL_COMPLETED) == 1
    assert presenter.codes().count(InitializationEventCode.INITIALIZATION_COMPLETED) == 1


def test_use_case_does_not_prompt_or_pull_when_disk_space_is_insufficient(tmp_path):
    ollama = FakeOllama(model_size=1024)
    use_case, presenter, _ = create_use_case(
        ollama,
        system=FakeSystem(free_bytes=1),
    )

    use_case.execute(tmp_path / "iacli")

    assert presenter.codes().count(InitializationEventCode.DISK_SPACE_INSUFFICIENT) == 1
    assert presenter.confirmations == []
    assert ollama.pulls == []


def test_use_case_reports_unreachable_ollama_but_completes_configuration(tmp_path):
    use_case, presenter, _ = create_use_case(
        FakeOllama(list_error=URLError("offline")),
    )

    use_case.execute(tmp_path / "iacli")

    assert presenter.codes().count(InitializationEventCode.OLLAMA_UNREACHABLE) == 1
    assert presenter.codes().count(InitializationEventCode.CONFIGURATION_CREATED) == 1
    assert presenter.codes().count(InitializationEventCode.INITIALIZATION_COMPLETED) == 1


def test_use_case_warns_for_missing_model_in_registry(tmp_path):
    use_case, presenter, _ = create_use_case(
        FakeOllama(size_error=HTTPError("url", 404, "Not Found", {}, None)),
    )

    use_case.execute(tmp_path / "iacli")

    assert presenter.codes().count(InitializationEventCode.MODEL_NOT_FOUND) == 1
    assert presenter.confirmations == []


def test_use_case_does_not_swallow_keyboard_interrupt_during_pull(tmp_path):
    use_case, presenter, _ = create_use_case(
        FakeOllama(pull_error=KeyboardInterrupt()),
        presenter=RecordingPresenter(confirm=True),
    )

    with pytest.raises(KeyboardInterrupt):
        use_case.execute(tmp_path / "iacli")

    assert InitializationEventCode.INITIALIZATION_COMPLETED not in presenter.codes()


def test_cli_presenter_maps_events_and_confirms_without_exposing_it_to_use_case():
    output = []
    answers = iter(("oui",))
    presenter = CliPresenter(input_fn=lambda _prompt: next(answers), output_fn=output.append)

    presenter.present(
        InitializationEvent(
            InitializationEventCode.PULL_PROGRESS,
            {"status": "downloading", "completed": 25 * 1024**2, "total": 100 * 1024**2},
        )
    )

    assert presenter.confirm_model_download("model", 1024) is True
    assert len(output) == 1


def test_cli_returns_interrupt_status_when_initialization_is_cancelled(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["iacli", "init"])
    monkeypatch.setattr(
        "iacli.cli.handle_init",
        lambda _args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    from iacli.cli import main

    assert main() == 130
    assert capsys.readouterr().err


def test_ollama_public_api_uses_http_gateway_and_parses_responses(fake_urlopen):
    client = OllamaClient(
        base_url="http://ollama.test:11434",
        registry_url="https://registry.test",
        urlopen=fake_urlopen,
    )
    fake_urlopen.add_response(
        client.list_models_url,
        b'{"models":[{"name":"local:latest"}, {"name":"other:tag"}]}',
    )
    fake_urlopen.add_response(
        client.model_manifest_url("remote:latest"),
        json.dumps({"config": {"size": 1}, "layers": [{"size": 9}]}).encode(),
    )
    fake_urlopen.add_response(
        client.pull_url("remote:latest"),
        b'{"status":"pulling"}\n{"status":"success"}\n',
    )
    progress = []

    assert client.installed_models() == {"local:latest", "other:tag"}
    assert client.remote_model_size("remote:latest") == 10
    client.pull_model("remote:latest", progress.append)

    assert len(fake_urlopen.requests) == 3
    assert json.loads(fake_urlopen.requests[-1][0].data) == {
        "name": "remote:latest",
        "stream": True,
    }
    assert [item.status for item in progress] == ["pulling", "success"]


def test_registry_index_selects_the_current_platform_via_public_api(
    fake_urlopen,
    monkeypatch,
):
    monkeypatch.setattr("iacli.infrastructure.ollama.platform.system", lambda: "Windows")
    monkeypatch.setattr("iacli.infrastructure.ollama.platform.machine", lambda: "AMD64")
    client = OllamaClient(registry_url="https://registry.test", urlopen=fake_urlopen)
    manifest_url = client.model_manifest_url("model:tag")
    fake_urlopen.add_response(
        manifest_url,
        json.dumps(
            {
                "manifests": [
                    {
                        "digest": "sha256:linux",
                        "platform": {"architecture": "amd64", "os": "linux"},
                    },
                    {
                        "digest": "sha256:windows",
                        "platform": {"architecture": "amd64", "os": "windows"},
                    },
                ]
            }
        ).encode(),
    )
    fake_urlopen.add_response(
        "https://registry.test/v2/library/model/manifests/sha256:windows",
        b'{"config":{"size":1},"layers":[{"size":2}]}',
    )

    assert client.remote_model_size("model:tag") == 3
    assert fake_urlopen.requests[1][0].full_url.endswith("sha256:windows")
