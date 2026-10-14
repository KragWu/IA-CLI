from argparse import Namespace
from io import StringIO
from pathlib import Path
import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.cli import build_parser, handle_chat
from iacli.domain.config import ConfigurationState, OllamaSettings
from iacli.services.chat import run_chat
from tests.fakes import FakeChatModel


class TtyStringIO(StringIO):
    def isatty(self):
        return True


def test_cli_without_subcommand_starts_chat():
    args = build_parser().parse_args([])

    assert args.func is handle_chat
    assert args.config_dir is None


def test_chat_command_routes_to_chat_handler():
    args = build_parser().parse_args(["chat", "--config-dir", "config"])

    assert args.func is handle_chat
    assert str(args.config_dir) == "config"


def test_chat_handler_uses_configured_model_and_instructions(tmp_path, monkeypatch):
    (tmp_path / "IACLI.md").write_text("Mes instructions", encoding="utf-8")
    settings = OllamaSettings(
        "http://ollama.test:11434",
        "https://registry.ollama.ai",
        "coder:latest",
    )
    state = ConfigurationState(tmp_path, settings)
    model = object()
    chat_calls = []
    model_arguments = {}

    class FakeGateway:
        def installed_models(self):
            return {"coder:latest"}

    class FakeConfigGateway:
        def ensure_configuration(self, directory):
            assert directory == tmp_path
            return state

    def create_client(_self, actual_settings):
        assert actual_settings == settings
        return FakeGateway()

    def create_chat_model(**kwargs):
        model_arguments.update(kwargs)
        return model

    monkeypatch.setattr(
        "iacli.cli.FileConfigurationGateway",
        FakeConfigGateway,
    )
    monkeypatch.setattr(
        "iacli.cli.OllamaClientFactory.create",
        create_client,
    )
    monkeypatch.setattr(
        "langchain_ollama.ChatOllama",
        create_chat_model,
    )
    monkeypatch.setattr(
        "iacli.cli.run_chat",
        lambda *args: chat_calls.append(args),
    )

    handle_chat(Namespace(config_dir=tmp_path))

    assert len(chat_calls) == 1
    assert chat_calls[0][0] is model
    assert chat_calls[0][1] == Path.cwd().resolve()
    assert chat_calls[0][2:] == ("Mes instructions", "coder:latest")
    assert model_arguments == {
        "model": "coder:latest",
        "base_url": "http://ollama.test:11434",
        "temperature": 0,
    }


def test_chat_runs_multiple_turns_and_exits_cleanly(tmp_path):
    model = FakeChatModel()
    prompts = iter(["première demande", "deuxième demande", "/exit"])
    output = []
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: next(prompts),
        output_fn=output.append,
    )

    run_chat(
        model,
        tmp_path,
        "Répondre en français.",
        "fake-model",
        presenter=presenter,
    )

    assert len(model.calls) == 2
    assert model.calls[1][1].content == "première demande"
    assert model.calls[1][-1].content == "deuxième demande"
    assert any("Réponse 1" in line for line in output)
    assert any("Réponse 2" in line for line in output)
    assert sum(line.startswith("temps action> modèle") for line in output) == 2
    assert sum(line.startswith("temps total> demande") for line in output) == 2


def test_chat_reports_tool_progress(tmp_path):
    (tmp_path / "notes.txt").write_text("contenu du projet", encoding="utf-8")

    class ToolCallingModel(FakeChatModel):
        def invoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(
                    content='{"name":"read_file","arguments":{"path":"notes.txt"}}'
                )
            assert isinstance(messages[-1], ToolMessage)
            assert messages[-1].content == "contenu du projet"
            return AIMessage(content="Le fichier contient le contenu du projet.")

    class RecordingPresenter(CliChatPresenter):
        def __init__(self):
            super().__init__(
                input_fn=lambda _prompt: "",
                output_fn=lambda _line: None,
            )
            self.progress_updates = []
            self.tool_events = []

        def prompt(self):
            return next(self.prompts)

        def start(self, workspace, model):
            self.prompts = iter(("Lis notes.txt", "/exit"))

        def start_progress(self):
            self.progress_updates.append("started")

        def update_progress(self, message):
            self.progress_updates.append(message)

        def stop_progress(self):
            self.progress_updates.append("stopped")

        def show_tool_call(self, name, arguments):
            self.tool_events.append(("call", name, arguments))

        def show_tool_result(self, name):
            self.tool_events.append(("result", name))

    model = ToolCallingModel()
    presenter = RecordingPresenter()
    run_chat(model, tmp_path, "", "fake-model", presenter=presenter)

    assert "Appel des outils : read_file" in presenter.progress_updates
    assert any("Résultat de read_file reçu" in item for item in presenter.progress_updates)
    assert presenter.progress_updates[0] == "started"
    assert presenter.progress_updates[-1] == "stopped"
    assert presenter.tool_events == [
        ("call", "read_file", {"path": "notes.txt"}),
        ("result", "read_file"),
    ]


def test_chat_executes_multiple_json_tool_calls_before_final_response(tmp_path):
    (tmp_path / "README.md").write_text("readme contents", encoding="utf-8")
    source_file = tmp_path / "src" / "iacli"
    source_file.mkdir(parents=True)
    (source_file / "cli.py").write_text("cli contents", encoding="utf-8")

    class MultiToolCallingModel(FakeChatModel):
        def invoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(
                    content=(
                        '{"name":"read_file","arguments":{"path":"README.md"}}\n'
                        '{"name":"read_file","arguments":{"path":"cli.py"}}'
                    )
                )
            tool_messages = [
                message for message in messages if isinstance(message, ToolMessage)
            ]
            assert [message.content for message in tool_messages] == [
                "readme contents",
                "cli contents",
            ]
            return AIMessage(content="Voici les premières lignes des deux fichiers.")

    model = MultiToolCallingModel()
    prompts = iter(
        (
            "Montre les premières lignes du README.md puis cli.py.",
            "/exit",
        )
    )
    output = []
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: next(prompts),
        output_fn=output.append,
    )

    run_chat(model, tmp_path, "", "fake-model", presenter=presenter)

    assert len(model.calls) == 2
    assert any("Voici les premières lignes des deux fichiers." in line for line in output)


def test_chat_exits_cleanly_on_end_of_input(tmp_path):
    def end_input(_prompt):
        raise EOFError

    presenter = CliChatPresenter(
        input_fn=end_input,
        output_fn=lambda _line: None,
    )

    run_chat(FakeChatModel(), tmp_path, "", "fake-model", presenter=presenter)


def test_progress_animation_displays_elapsed_time():
    stream = TtyStringIO()
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: "",
        output_fn=lambda _line: None,
        progress_stream=stream,
    )

    presenter.start_progress()
    time.sleep(0.15)
    presenter.stop_progress()

    assert "L'agent traite votre demande (" in stream.getvalue()
    assert " ms)..." in stream.getvalue() or " s)..." in stream.getvalue()


def test_duration_formatter_uses_milliseconds_seconds_minutes_and_hours():
    format_duration = CliChatPresenter._format_duration

    assert format_duration(0.5) == "500 ms"
    assert format_duration(1.25) == "1.25 s"
    assert format_duration(60) == "1.00 min"
    assert format_duration(3600) == "1.00 h"
