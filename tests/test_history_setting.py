"""Historique de saisie : désactivé par défaut, activé par `[chat] save_history`."""

import os
import stat
import sys
from argparse import Namespace

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from iacli import cli
from iacli.adapters.terminal_input import create_prompt_input
from iacli.domain.commands import COMMANDS
from iacli.domain.config import (
    ChatSettings,
    ConfigurationState,
    OllamaSettings,
)
from iacli.domain.models import ModelInfo
from iacli.domain.status import PromptStatus
from iacli.infrastructure.configuration import DEFAULT_CONFIG, FileConfigurationGateway


def load_configuration(tmp_path, toml):
    directory = tmp_path / "iacli"
    directory.mkdir()
    (directory / "config.toml").write_text(toml, encoding="utf-8")
    return FileConfigurationGateway().ensure_configuration(directory)


def test_the_history_is_not_saved_by_default(tmp_path):
    state = load_configuration(tmp_path, '[ollama]\nmodel = "custom:latest"\n')

    assert state.chat == ChatSettings(save_history=False)
    assert state.warnings == ()


def test_the_default_configuration_documents_the_history_setting(tmp_path):
    assert "save_history" in DEFAULT_CONFIG

    directory = tmp_path / "neuf"
    state = FileConfigurationGateway().ensure_configuration(directory)

    assert state.chat == ChatSettings(save_history=False)


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_save_history_is_read_from_the_chat_section(tmp_path, value, expected):
    state = load_configuration(tmp_path, f"[chat]\nsave_history = {value}\n")

    assert state.chat == ChatSettings(save_history=expected)
    assert state.warnings == ()


@pytest.mark.parametrize(
    "toml",
    [
        '[chat]\nsave_history = "oui"\n',
        "[chat]\nsave_history = 1\n",
        'chat = "texte"\n',
    ],
)
def test_invalid_history_setting_warns_and_keeps_the_history_off(tmp_path, toml):
    state = load_configuration(tmp_path, toml)

    assert state.chat == ChatSettings(save_history=False)
    assert any("chat" in warning for warning in state.warnings)


def test_no_history_file_is_created_without_a_history_path(tmp_path):
    with create_pipe_input() as pipe:
        read = create_prompt_input(
            COMMANDS,
            history_path=None,
            status=lambda: PromptStatus("m", None, None),
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text("secret\r")

        assert read("vous> ") == "secret"

    assert list(tmp_path.iterdir()) == []


def test_an_unusable_history_path_degrades_to_no_history(tmp_path):
    blocker = tmp_path / "bloqueur"
    blocker.write_text("un fichier, pas un dossier", encoding="utf-8")

    with create_pipe_input() as pipe:
        read = create_prompt_input(
            COMMANDS,
            history_path=blocker / "history",
            status=lambda: PromptStatus("m", None, None),
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text("bonjour\r")

        assert read("vous> ") == "bonjour"


def test_a_history_file_that_cannot_be_read_or_written_does_not_break_the_input(
    tmp_path,
):
    # Un dossier à la place du fichier : l'ouverture échoue en lecture comme en écriture.
    with create_pipe_input() as pipe:
        read = create_prompt_input(
            COMMANDS,
            history_path=tmp_path,
            status=lambda: PromptStatus("m", None, None),
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text("bonjour\r")

        assert read("vous> ") == "bonjour"


def test_the_chat_presenter_gets_no_history_path_when_none_is_given(monkeypatch):
    created = {}

    class Tty:
        def isatty(self):
            return True

    def fake_create_prompt_input(commands, history_path, status):
        created["history_path"] = history_path
        return lambda prompt: ""

    monkeypatch.setattr(cli.sys, "stdin", Tty())
    monkeypatch.setattr(cli.sys, "stdout", Tty())
    monkeypatch.setattr(cli, "create_prompt_input", fake_create_prompt_input)

    cli.create_chat_presenter(None)

    assert created["history_path"] is None


@pytest.mark.parametrize(
    ("save_history", "expected"), [(False, None), (True, "history")]
)
def test_handle_chat_saves_the_history_only_when_enabled(
    tmp_path, monkeypatch, save_history, expected
):
    (tmp_path / "IACLI.md").write_text("", encoding="utf-8")
    settings = OllamaSettings(
        "http://ollama.test:11434", "https://registry.ollama.ai", "coder:latest"
    )
    state = ConfigurationState(tmp_path, settings, chat=ChatSettings(save_history))
    received = []

    class FakeGateway:
        def installed_models(self):
            return {"coder:latest"}

        def model_info(self, model):
            return ModelInfo(capabilities=frozenset({"completion", "tools"}))

    class FakeConfigGateway:
        def ensure_configuration(self, directory):
            return state

    monkeypatch.setattr("iacli.cli.FileConfigurationGateway", FakeConfigGateway)
    monkeypatch.setattr(
        "iacli.cli.OllamaClientFactory.create", lambda _self, _s: FakeGateway()
    )
    monkeypatch.setattr("langchain_ollama.ChatOllama", lambda **kwargs: object())
    monkeypatch.setattr(
        cli, "create_chat_presenter", lambda history: received.append(history)
    )
    monkeypatch.setattr("iacli.cli.run_chat", lambda *args, **kwargs: None)

    cli.handle_chat(Namespace(config_dir=tmp_path))

    assert received == [None if expected is None else tmp_path / expected]


posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="les droits POSIX n'existent pas sous Windows"
)


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@posix_only
def test_a_new_history_file_is_private_to_its_owner(tmp_path):
    history = tmp_path / "config" / "history"
    previous = os.umask(0o022)
    try:
        with create_pipe_input() as pipe:
            read = create_prompt_input(
                COMMANDS,
                history_path=history,
                status=lambda: PromptStatus("m", None, None),
                input=pipe,
                output=DummyOutput(),
            )
            pipe.send_text("jeton secret" + chr(13))
            read("vous> ")
    finally:
        os.umask(previous)

    assert _mode(history) == 0o600
    assert "jeton secret" in history.read_text(encoding="utf-8")


@posix_only
def test_an_existing_world_readable_history_file_is_made_private(tmp_path):
    history = tmp_path / "history"
    history.write_text("ancien", encoding="utf-8")
    history.chmod(0o644)

    create_prompt_input(
        COMMANDS,
        history_path=history,
        status=lambda: PromptStatus("m", None, None),
        input=None,
        output=DummyOutput(),
    )

    assert _mode(history) == 0o600
