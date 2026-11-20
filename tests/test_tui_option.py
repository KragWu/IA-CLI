import io
import sys
from argparse import Namespace

import pytest

from iacli import cli
from iacli.cli import build_parser, handle_chat
from iacli.domain.config import (
    ChatSettings,
    ConfigurationState,
    FileSettings,
    OllamaSettings,
)
from iacli.domain.models import ModelInfo
from tests.fakes import FakeChatModel


class _Tty(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("argv", [["--tui"], ["chat", "--tui"], ["--tui", "chat"]])
def test_the_tui_option_is_accepted_with_or_without_the_chat_command(argv):
    assert build_parser().parse_args(argv).tui is True


def test_the_tui_option_is_off_by_default():
    assert build_parser().parse_args([]).tui is False
    assert build_parser().parse_args(["chat"]).tui is False


@pytest.fixture
def chat_environment(request, tmp_path, monkeypatch):
    (tmp_path / "IACLI.md").write_text("instructions", encoding="utf-8")
    settings = OllamaSettings("http://x", "http://y", "m:latest", num_ctx=4096)
    chat = ChatSettings(save_history=getattr(request, "param", False))
    state = ConfigurationState(tmp_path, settings, files=FileSettings(), chat=chat)

    class Gateway:
        def installed_models(self):
            return {"m:latest"}

        def model_info(self, model):
            return ModelInfo(capabilities=frozenset({"tools"}))

    class Config:
        def ensure_configuration(self, directory):
            return state

    monkeypatch.setattr("iacli.cli.FileConfigurationGateway", Config)
    monkeypatch.setattr(
        "iacli.cli.OllamaClientFactory.create", lambda _self, _settings: Gateway()
    )
    monkeypatch.setattr("langchain_ollama.ChatOllama", lambda **kwargs: FakeChatModel())
    return tmp_path


def test_tui_runs_the_textual_app_instead_of_the_console_loop(
    chat_environment, monkeypatch
):
    started = []
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(
        "iacli.cli.run_chat",
        lambda *a, **k: pytest.fail("la boucle console ne doit pas démarrer"),
    )
    monkeypatch.setattr(cli, "run_tui", lambda app: started.append(app))

    handle_chat(Namespace(config_dir=chat_environment, tui=True))

    (app,) = started
    assert type(app).__name__ == "ChatApp"


def test_tui_is_refused_outside_an_interactive_terminal(chat_environment, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(cli, "run_tui", lambda app: pytest.fail("pas de terminal"))

    with pytest.raises(RuntimeError, match="--tui"):
        handle_chat(Namespace(config_dir=chat_environment, tui=True))


def test_the_console_loop_is_still_used_without_the_option(
    chat_environment, monkeypatch
):
    calls = []
    monkeypatch.setattr("iacli.cli.run_chat", lambda *a, **k: calls.append(1))

    handle_chat(Namespace(config_dir=chat_environment))

    assert calls == [1]


def test_the_tui_renders_responses_as_markdown(chat_environment, monkeypatch):
    started = []
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(cli, "run_tui", lambda app: started.append(app))

    handle_chat(Namespace(config_dir=chat_environment, tui=True))

    (app,) = started
    lines = []
    app._presenter._emit = lines.append
    app._presenter.show_response("**gras**")

    assert chr(27) + "[1m" in "".join(lines)


def test_the_tui_colors_the_diffs(chat_environment, monkeypatch):
    started = []
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(cli, "run_tui", lambda app: started.append(app))

    handle_chat(Namespace(config_dir=chat_environment, tui=True))

    (app,) = started
    lines = []
    app._presenter._emit = lines.append
    app._presenter._ask = lambda question: True
    app._presenter._screen = None  # application non montée
    app._presenter.confirm_file_write("a.txt", "+ajout")

    assert chr(27) + "[" in "".join(lines)


def test_the_tui_leaves_the_mouse_to_the_terminal_so_the_text_can_be_selected():
    calls = []

    class App:
        def run(self, **options):
            calls.append(options)

    cli.run_tui(App())

    assert calls == [{"inline": True, "mouse": False}]


def _started_app(chat_environment, monkeypatch):
    started = []
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(cli, "run_tui", lambda app: started.append(app))
    handle_chat(Namespace(config_dir=chat_environment, tui=True))
    return started[0]


@pytest.mark.parametrize("chat_environment", [True], indirect=True)
def test_the_tui_saves_the_history_in_the_configuration_folder_when_enabled(
    chat_environment, monkeypatch
):
    app = _started_app(chat_environment, monkeypatch)

    app._history_store.append("bonjour")

    assert (chat_environment / "history").exists()


def test_the_tui_keeps_the_history_in_memory_by_default(chat_environment, monkeypatch):
    app = _started_app(chat_environment, monkeypatch)

    assert app._history_store is None
