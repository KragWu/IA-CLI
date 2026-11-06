from argparse import Namespace
from io import StringIO
from pathlib import Path
import time

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.cli import build_parser, handle_chat
from iacli.infrastructure.workspace_tools import create_workspace_tools
from iacli.domain.config import ConfigurationState, OllamaSettings
from iacli.domain.models import ModelInfo
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
        num_ctx=8192,
    )
    state = ConfigurationState(tmp_path, settings)
    model = object()
    chat_calls = []
    model_arguments = {}

    class FakeGateway:
        def installed_models(self):
            return {"coder:latest"}

        def model_info(self, model):
            return ModelInfo(capabilities=frozenset({"completion", "tools"}))

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
        lambda *args, **kwargs: chat_calls.append((args, kwargs)),
    )

    handle_chat(Namespace(config_dir=tmp_path))

    assert len(chat_calls) == 1
    args, kwargs = chat_calls[0]
    assert args[0] is model
    assert args[1] == Path.cwd().resolve()
    assert args[2:] == ("Mes instructions", "coder:latest")
    assert isinstance(kwargs["presenter"], CliChatPresenter)
    assert kwargs["tools_factory"] is create_workspace_tools
    assert kwargs["context_limit"] == 8192
    assert model_arguments == {
        "model": "coder:latest",
        "base_url": "http://ollama.test:11434",
        "temperature": 0,
        "num_ctx": 8192,
    }


def test_chat_builds_tools_through_injected_factory_with_presenter_callbacks(
    tmp_path,
):
    created = []

    def tools_factory(workspace, confirm_write, confirm_shell, on_timing):
        created.append((workspace, confirm_write, confirm_shell, on_timing))
        return []

    prompts = iter(["/exit"])
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: next(prompts),
        output_fn=lambda _line: None,
    )

    run_chat(
        FakeChatModel(),
        tmp_path,
        "",
        "fake-model",
        presenter=presenter,
        tools_factory=tools_factory,
    )

    assert created == [
        (
            tmp_path,
            presenter.confirm_file_write,
            presenter.confirm_shell_command,
            presenter.show_action_timing,
        )
    ]


def _handle_chat_with_capabilities(
    tmp_path,
    monkeypatch,
    capabilities,
    *,
    context_length=None,
    num_ctx=18000,
    model_arguments=None,
    chat_kwargs=None,
):
    (tmp_path / "IACLI.md").write_text("Mes instructions", encoding="utf-8")
    settings = OllamaSettings(
        "http://ollama.test:11434",
        "https://registry.ollama.ai",
        "coder:latest",
        num_ctx=num_ctx,
    )
    chat_calls = []

    class FakeGateway:
        def installed_models(self):
            return {"coder:latest"}

        def model_info(self, model):
            assert model == "coder:latest"
            return ModelInfo(
                capabilities=None if capabilities is None else frozenset(capabilities),
                context_length=context_length,
            )

    class FakeConfigGateway:
        def ensure_configuration(self, directory):
            return ConfigurationState(tmp_path, settings)

    def create_chat_model(**kwargs):
        if model_arguments is not None:
            model_arguments.update(kwargs)
        return object()

    monkeypatch.setattr("iacli.cli.FileConfigurationGateway", FakeConfigGateway)
    monkeypatch.setattr(
        "iacli.cli.OllamaClientFactory.create", lambda _self, _settings: FakeGateway()
    )
    monkeypatch.setattr("langchain_ollama.ChatOllama", create_chat_model)
    def fake_run_chat(*args, **kwargs):
        chat_calls.append(args)
        if chat_kwargs is not None:
            chat_kwargs.update(kwargs)

    monkeypatch.setattr("iacli.cli.run_chat", fake_run_chat)

    handle_chat(Namespace(config_dir=tmp_path))
    return chat_calls


def test_chat_handler_refuses_model_without_tool_support(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="outils"):
        _handle_chat_with_capabilities(tmp_path, monkeypatch, {"completion"})


def test_chat_handler_starts_when_model_supports_tools(tmp_path, monkeypatch):
    calls = _handle_chat_with_capabilities(
        tmp_path, monkeypatch, {"completion", "tools"}
    )

    assert len(calls) == 1


def test_chat_handler_starts_when_capabilities_are_unknown(tmp_path, monkeypatch):
    calls = _handle_chat_with_capabilities(tmp_path, monkeypatch, None)

    assert len(calls) == 1


def test_chat_handler_caps_num_ctx_to_the_model_context_length(
    tmp_path, monkeypatch, capsys
):
    arguments = {}
    chat_kwargs = {}

    _handle_chat_with_capabilities(
        tmp_path,
        monkeypatch,
        {"tools"},
        context_length=8192,
        num_ctx=18000,
        model_arguments=arguments,
        chat_kwargs=chat_kwargs,
    )

    assert arguments["num_ctx"] == 8192
    assert chat_kwargs["context_limit"] == 8192
    warning = capsys.readouterr().err
    assert warning.startswith("[iacli] Avertissement :")
    assert "18000" in warning
    assert "8192" in warning


def test_chat_handler_keeps_num_ctx_when_within_the_model_context_length(
    tmp_path, monkeypatch, capsys
):
    arguments = {}

    _handle_chat_with_capabilities(
        tmp_path,
        monkeypatch,
        {"tools"},
        context_length=32768,
        num_ctx=18000,
        model_arguments=arguments,
    )

    assert arguments["num_ctx"] == 18000
    assert capsys.readouterr().err == ""


def test_chat_handler_keeps_num_ctx_when_model_context_length_is_unknown(
    tmp_path, monkeypatch, capsys
):
    arguments = {}

    _handle_chat_with_capabilities(
        tmp_path, monkeypatch, {"tools"}, num_ctx=18000, model_arguments=arguments
    )

    assert arguments["num_ctx"] == 18000
    assert capsys.readouterr().err == ""


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
        tools_factory=create_workspace_tools,
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
    run_chat(model, tmp_path, "", "fake-model", presenter=presenter, tools_factory=create_workspace_tools)

    assert "Appel des outils : read_file" in presenter.progress_updates
    assert any("Résultat de read_file reçu" in item for item in presenter.progress_updates)
    assert presenter.progress_updates[0] == "started"
    assert presenter.progress_updates[-1] == "stopped"
    assert presenter.tool_events == [
        ("call", "read_file", {"path": "notes.txt"}),
        ("result", "read_file"),
    ]


def test_chat_reports_turn_error_and_continues_session(tmp_path):
    class FlakyModel(FakeChatModel):
        def invoke(self, messages):
            if not self.calls:
                self.calls.append(messages)
                raise ConnectionError("Ollama est injoignable")
            return super().invoke(messages)

    model = FlakyModel()
    prompts = iter(["demande en échec", "demande suivante", "/exit"])
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
        tools_factory=create_workspace_tools,
    )

    assert any(
        line.startswith("erreur>") and "Ollama est injoignable" in line
        for line in output
    )
    assert any("Réponse 2" in line for line in output)
    assert [message.content for message in model.calls[1] if isinstance(message, HumanMessage)] == [
        "demande en échec",
        "demande suivante",
    ]
    assert sum(line.startswith("temps total> demande") for line in output) == 2


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

    run_chat(model, tmp_path, "", "fake-model", presenter=presenter, tools_factory=create_workspace_tools)

    assert len(model.calls) == 2
    assert any("Voici les premières lignes des deux fichiers." in line for line in output)


def test_chat_reports_every_tool_result_when_several_tools_run_in_one_step(tmp_path):
    (tmp_path / "README.md").write_text("readme contents", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("notes contents", encoding="utf-8")

    class MultiToolCallingModel(FakeChatModel):
        def invoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(
                    content=(
                        '{"name":"read_file","arguments":{"path":"README.md"}}\n'
                        '{"name":"list_files","arguments":{"path":"."}}'
                    )
                )
            return AIMessage(content="Terminé.")

    class RecordingPresenter(CliChatPresenter):
        def __init__(self):
            prompts = iter(["Lis puis liste.", "/exit"])
            super().__init__(
                input_fn=lambda _prompt: next(prompts),
                output_fn=lambda _line: None,
            )
            self.tool_results = []

        def show_tool_result(self, name):
            self.tool_results.append(name)
            super().show_tool_result(name)

    presenter = RecordingPresenter()

    run_chat(
        MultiToolCallingModel(),
        tmp_path,
        "",
        "fake-model",
        presenter=presenter,
        tools_factory=create_workspace_tools,
    )

    assert presenter.tool_results == ["read_file", "list_files"]


def test_chat_exits_cleanly_on_end_of_input(tmp_path):
    def end_input(_prompt):
        raise EOFError

    presenter = CliChatPresenter(
        input_fn=end_input,
        output_fn=lambda _line: None,
    )

    run_chat(FakeChatModel(), tmp_path, "", "fake-model", presenter=presenter, tools_factory=create_workspace_tools)


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
    assert format_duration(60) == "1 min 00 s"
    assert format_duration(251) == "4 min 11 s"
    assert format_duration(3600) == "1 h 00 min 00 s"
    assert format_duration(3661) == "1 h 01 min 01 s"


def test_output_clears_spinner_line_before_printing_while_progress_is_running():
    stream = TtyStringIO()
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: "",
        output_fn=lambda line: stream.write(f"{line}\n"),
        progress_stream=stream,
    )

    presenter.start_progress()
    time.sleep(0.15)
    for _ in range(20):
        presenter.show_tool_call("read_file", {"path": "a.py"})
        presenter.show_action_timing("read_file", 0.01)
        time.sleep(0.01)
    presenter.stop_progress()

    written = stream.getvalue()
    assert written.count("outil> read_file") == 20
    assert written.count("\r\033[2Koutil> read_file") == 20
    assert written.count("\r\033[2Ktemps action> read_file") == 20


def test_spinner_does_not_draw_while_an_output_line_is_being_printed():
    log = []

    class LoggingStream(TtyStringIO):
        def write(self, text):
            log.append(("spinner", text))
            return super().write(text)

    def slow_output(line):
        log.append(("out-start", line))
        time.sleep(0.35)
        log.append(("out-end", line))

    presenter = CliChatPresenter(
        input_fn=lambda _prompt: "",
        output_fn=slow_output,
        progress_stream=LoggingStream(),
    )
    presenter.start_progress()
    time.sleep(0.15)
    log.clear()

    presenter.show_error("boom")
    presenter.stop_progress()

    kinds = [kind for kind, _text in log]
    start = kinds.index("out-start")
    end = kinds.index("out-end")
    assert "spinner" not in kinds[start:end]


def _run_two_turns(tmp_path, responses, monkeypatch=None, max_iterations=None):
    (tmp_path / "notes.txt").write_text("contenu du projet", encoding="utf-8")
    if max_iterations is not None:
        monkeypatch.setattr("iacli.services.chat.MAX_TOOL_ITERATIONS", max_iterations)
    model = FakeChatModel(responses)
    prompts = iter(["première demande", "seconde demande", "/exit"])
    output = []
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: next(prompts),
        output_fn=output.append,
    )

    run_chat(
        model,
        tmp_path,
        "",
        "fake-model",
        presenter=presenter,
        tools_factory=create_workspace_tools,
    )
    return model, output


READ_NOTES = AIMessage(content='{"name":"read_file","arguments":{"path":"notes.txt"}}')


def test_failed_turn_keeps_completed_tool_work_in_history(tmp_path):
    model, output = _run_two_turns(
        tmp_path,
        [READ_NOTES, AIMessage(content=""), AIMessage(content="Réponse finale.")],
    )

    history = model.calls[2]
    assert [type(message) for message in history] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        HumanMessage,
    ]
    assert history[1].content == "première demande"
    assert history[2].tool_calls[0]["name"] == "read_file"
    assert history[3].content == "contenu du projet"
    assert history[4].content == "seconde demande"
    assert any(line.startswith("erreur>") for line in output)
    assert any("Réponse finale." in line for line in output)


def test_failed_turn_drops_tool_call_without_result_from_history(
    tmp_path, monkeypatch
):
    model, output = _run_two_turns(
        tmp_path,
        [READ_NOTES, READ_NOTES, AIMessage(content="Réponse finale.")],
        monkeypatch,
        max_iterations=1,
    )

    history = model.calls[2]
    assert [type(message) for message in history] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        HumanMessage,
    ]
    assert any(line.startswith("erreur>") and "Limite" in line for line in output)


def test_chat_reports_error_when_graph_produces_no_state(tmp_path, monkeypatch):
    class EmptyGraph:
        def stream(self, *_args, **_kwargs):
            return iter(())

    monkeypatch.setattr(
        "iacli.services.chat.build_control_graph", lambda *_args, **_kwargs: EmptyGraph()
    )
    prompts = iter(["demande", "/exit"])
    output = []
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: next(prompts),
        output_fn=output.append,
    )

    run_chat(
        FakeChatModel(),
        tmp_path,
        "",
        "fake-model",
        presenter=presenter,
        tools_factory=create_workspace_tools,
    )

    assert any(line.startswith("erreur>") and "aucun état" in line for line in output)
    assert not any(line.startswith("\niacli>") for line in output)


def test_ctrl_c_during_a_turn_cancels_only_that_turn(tmp_path):
    class InterruptedModel(FakeChatModel):
        def invoke(self, messages):
            if not self.calls:
                self.calls.append(messages)
                raise KeyboardInterrupt
            return super().invoke(messages)

    model = InterruptedModel()
    prompts = iter(["demande lente", "demande suivante", "/exit"])
    output = []
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: next(prompts),
        output_fn=output.append,
    )

    run_chat(
        model,
        tmp_path,
        "",
        "fake-model",
        presenter=presenter,
        tools_factory=create_workspace_tools,
    )

    assert any(line.startswith("erreur>") and "interrompu" in line for line in output)
    assert any("Réponse 2" in line for line in output)
    assert [
        message.content for message in model.calls[1] if isinstance(message, HumanMessage)
    ] == ["demande lente", "demande suivante"]
    assert sum(line.startswith("temps total> demande") for line in output) == 2


def test_ctrl_c_at_the_prompt_ends_the_session_cleanly(tmp_path):
    def interrupted_input(_prompt):
        raise KeyboardInterrupt

    presenter = CliChatPresenter(
        input_fn=interrupted_input,
        output_fn=lambda _line: None,
    )

    run_chat(
        FakeChatModel(),
        tmp_path,
        "",
        "fake-model",
        presenter=presenter,
        tools_factory=create_workspace_tools,
    )


def _usage(total):
    return {"input_tokens": total - 100, "output_tokens": 100, "total_tokens": total}


class ContextRecordingPresenter(CliChatPresenter):
    def __init__(self, prompts):
        iterator = iter(prompts)
        super().__init__(
            input_fn=lambda _prompt: next(iterator),
            output_fn=lambda _line: None,
        )
        self.usages = []
        self.warnings = []

    def show_context_usage(self, used, limit):
        self.usages.append((used, limit))

    def show_context_warning(self, used, limit):
        self.warnings.append((used, limit))


def _run_with_usage(tmp_path, responses, *, context_limit=18000):
    presenter = ContextRecordingPresenter(["demande", "/exit"])
    run_chat(
        FakeChatModel(responses),
        tmp_path,
        "",
        "fake-model",
        presenter=presenter,
        tools_factory=create_workspace_tools,
        context_limit=context_limit,
    )
    return presenter


def test_chat_reports_context_usage_after_a_turn(tmp_path):
    presenter = _run_with_usage(
        tmp_path, [AIMessage(content="Réponse.", usage_metadata=_usage(4500))]
    )

    assert presenter.usages == [(4500, 18000)]
    assert presenter.warnings == []


def test_chat_warns_when_context_reaches_80_percent(tmp_path):
    presenter = _run_with_usage(
        tmp_path, [AIMessage(content="Réponse.", usage_metadata=_usage(14400))]
    )

    assert presenter.usages == [(14400, 18000)]
    assert presenter.warnings == [(14400, 18000)]


def test_chat_does_not_warn_just_below_80_percent(tmp_path):
    presenter = _run_with_usage(
        tmp_path, [AIMessage(content="Réponse.", usage_metadata=_usage(14399))]
    )

    assert presenter.warnings == []


def test_chat_uses_usage_of_the_last_model_call_of_the_turn(tmp_path):
    (tmp_path / "notes.txt").write_text("contenu", encoding="utf-8")
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"path": "notes.txt"},
                "id": "read-1",
                "type": "tool_call",
            }
        ],
        usage_metadata=_usage(3000),
    )

    presenter = _run_with_usage(
        tmp_path, [tool_call, AIMessage(content="Fini.", usage_metadata=_usage(5000))]
    )

    assert presenter.usages == [(5000, 18000)]


def test_chat_reports_nothing_when_usage_is_unknown(tmp_path):
    presenter = _run_with_usage(tmp_path, [AIMessage(content="Réponse.")])

    assert presenter.usages == []
    assert presenter.warnings == []


def test_chat_reports_nothing_without_a_context_limit(tmp_path):
    presenter = _run_with_usage(
        tmp_path,
        [AIMessage(content="Réponse.", usage_metadata=_usage(4500))],
        context_limit=None,
    )

    assert presenter.usages == []


def test_presenter_prints_context_usage_and_warning():
    output = []
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: "", output_fn=output.append
    )

    presenter.show_context_usage(4500, 18000)
    presenter.show_context_warning(14400, 18000)

    assert output[0] == "contexte> 4 500 / 18 000 tokens (25 %)"
    assert output[1].startswith("attention> ")
    assert "80 %" in output[1]
    assert "18 000" in output[1]
