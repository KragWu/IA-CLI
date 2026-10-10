from argparse import Namespace
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.cli import build_parser, handle_chat
from iacli.domain.config import ConfigurationState, OllamaSettings
from iacli.services.chat import run_chat


class FakeChatModel:
    def __init__(self) -> None:
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=f"Réponse {len(self.calls)}")


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


def test_chat_exits_cleanly_on_end_of_input(tmp_path):
    def end_input(_prompt):
        raise EOFError

    presenter = CliChatPresenter(
        input_fn=end_input,
        output_fn=lambda _line: None,
    )

    run_chat(FakeChatModel(), tmp_path, "", "fake-model", presenter=presenter)
