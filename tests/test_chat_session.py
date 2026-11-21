import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.infrastructure.workspace_tools import create_workspace_tools
from iacli.services.chat import ChatSession
from tests.fakes import FakeChatModel

TOOL_CALL = AIMessage(
    content="", tool_calls=[{"name": "list_files", "args": {}, "id": "1"}]
)


def _no_prompt(_text):
    raise AssertionError("la session ne doit pas lire la saisie elle-même")


def _session(tmp_path, model=None, context_limit=None):
    output = []
    presenter = CliChatPresenter(input_fn=_no_prompt, output_fn=output.append)
    session = ChatSession(
        model or FakeChatModel(),
        tmp_path,
        "",
        "fake-model",
        presenter=presenter,
        tools_factory=create_workspace_tools,
        context_limit=context_limit,
    )
    return session, output


def test_a_session_handles_messages_without_reading_the_input_itself(tmp_path):
    session, output = _session(tmp_path)

    assert session.handle("bonjour") is True

    assert any("Réponse 1" in line for line in output)


def test_handle_asks_to_stop_on_exit_commands_only(tmp_path):
    session, _ = _session(tmp_path)

    assert session.handle("/exit") is False
    assert session.handle("/QUIT") is False
    assert session.handle("/help") is True
    assert session.handle("   ") is True


def test_handle_ignores_blank_input_without_calling_the_model(tmp_path):
    model = FakeChatModel()
    session, _ = _session(tmp_path, model)

    session.handle("   ")

    assert model.calls == []


def test_the_history_is_kept_between_two_handled_messages(tmp_path):
    model = FakeChatModel()
    session, _ = _session(tmp_path, model)

    session.handle("premier")
    session.handle("second")

    assert [m.content for m in model.calls[1] if isinstance(m, HumanMessage)] == [
        "premier",
        "second",
    ]


def test_cancel_from_another_thread_stops_the_turn_between_two_steps(tmp_path):
    release = threading.Event()

    class SlowModel(FakeChatModel):
        def invoke(self, messages):
            self.calls.append(messages)
            release.wait(5)
            return TOOL_CALL

    model = SlowModel()
    session, output = _session(tmp_path, model)

    def cancel_later():
        session.cancel()
        release.set()

    threading.Timer(0.2, cancel_later).start()
    assert session.handle("demande longue") is True

    assert "Génération annulée" in output
    assert not any(line.startswith("erreur>") for line in output)
    assert len(model.calls) == 1
    assert not any(line.startswith("outil terminé>") for line in output)


def test_a_cancelled_turn_leaves_no_trace_in_the_history(tmp_path):
    holder = {}

    class CancellingModel(FakeChatModel):
        def invoke(self, messages):
            if not self.calls:
                self.calls.append(messages)
                holder["session"].cancel()
                return TOOL_CALL
            return super().invoke(messages)

    model = CancellingModel()
    session, _ = _session(tmp_path, model)
    holder["session"] = session

    session.handle("annulée")
    session.handle("suivante")

    assert [m.content for m in model.calls[1] if isinstance(m, HumanMessage)] == [
        "suivante"
    ]


def test_cancel_while_idle_does_not_cancel_the_next_turn(tmp_path):
    session, output = _session(tmp_path)

    session.cancel()
    session.handle("bonjour")

    assert "Génération annulée" not in output
    assert any("Réponse 1" in line for line in output)


def test_cancel_during_the_last_step_keeps_the_complete_answer(tmp_path):
    holder = {}

    class LastStepModel(FakeChatModel):
        def invoke(self, messages):
            self.calls.append(messages)
            holder["session"].cancel()
            return AIMessage(content=f"Réponse complète {len(self.calls)}")

    model = LastStepModel()
    session, output = _session(tmp_path, model)
    holder["session"] = session

    session.handle("question")
    session.handle("suite")

    assert any("Réponse complète 1" in line for line in output)
    assert "Génération annulée" not in output
    # La réponse terminée fait partie de l'historique du tour suivant.
    assert [m.content for m in model.calls[1] if isinstance(m, AIMessage)] == [
        "Réponse complète 1"
    ]


def test_cancel_still_stops_a_turn_that_is_retrying_an_unreadable_tool_call(tmp_path):
    holder = {}

    class UnreadableCallModel(FakeChatModel):
        def invoke(self, messages):
            self.calls.append(messages)
            holder["session"].cancel()
            return AIMessage(content='<tool_call>{"name": ')

    model = UnreadableCallModel()
    session, output = _session(tmp_path, model)
    holder["session"] = session

    session.handle("question")

    assert "Génération annulée" in output
    assert len(model.calls) == 1


def test_the_message_sent_to_the_model_keeps_its_indentation(tmp_path):
    model = FakeChatModel()
    session, _ = _session(tmp_path, model)
    snippet = "    def f():\n        return 1"

    session.handle(snippet)

    (sent,) = [m for m in model.calls[0] if isinstance(m, HumanMessage)]
    assert sent.content == snippet


@pytest.mark.parametrize(
    ("typed", "sent"),
    [
        ("question\n", "question"),
        ("question  \n\n", "question"),
        ("\n  \n    def f():\n        pass\n\n", "    def f():\n        pass"),
        ("un\n\ndeux", "un\n\ndeux"),
        ("  indenté", "  indenté"),
    ],
)
def test_only_surrounding_blank_lines_and_trailing_spaces_are_removed(
    tmp_path, typed, sent
):
    model = FakeChatModel()
    session, _ = _session(tmp_path, model)

    session.handle(typed)

    (message,) = [m for m in model.calls[0] if isinstance(m, HumanMessage)]
    assert message.content == sent


def test_a_command_is_still_recognised_around_whitespace(tmp_path):
    session, output = _session(tmp_path)

    assert session.handle("  /exit  ") is False
    assert session.handle("\n/help\n") is True
    assert "Commandes disponibles :" in output


def test_a_cancel_requested_while_the_input_is_being_parsed_is_not_lost(
    tmp_path, monkeypatch
):
    from iacli.services import chat as chat_module

    holder = {}
    real_parse_input = chat_module.parse_input

    def parse_then_cancel(text):
        holder["session"].cancel()
        return real_parse_input(text)

    monkeypatch.setattr(chat_module, "parse_input", parse_then_cancel)
    model = FakeChatModel()
    session, output = _session(tmp_path, model)
    holder["session"] = session

    session.handle("demande")

    assert "Génération annulée" in output
    assert not any("Réponse" in line for line in output)


def test_a_cancel_after_a_finished_message_does_not_affect_the_next_one(tmp_path):
    session, output = _session(tmp_path)

    session.handle("premier")
    session.cancel()
    session.handle("second")

    assert "Génération annulée" not in output
    assert any("Réponse 2" in line for line in output)


def test_a_cancel_during_a_command_does_not_affect_the_next_message(tmp_path):
    holder = {}

    class CancellingPresenter(CliChatPresenter):
        def show_help(self, commands):
            holder["session"].cancel()
            super().show_help(commands)

    output = []
    session = ChatSession(
        FakeChatModel(),
        tmp_path,
        "",
        "fake-model",
        presenter=CancellingPresenter(input_fn=_no_prompt, output_fn=output.append),
        tools_factory=create_workspace_tools,
    )
    holder["session"] = session

    session.handle("/help")
    session.handle("bonjour")

    assert "Génération annulée" not in output
    assert any("Réponse 1" in line for line in output)


def test_a_cancelled_turn_keeps_the_tools_already_run_in_the_history(tmp_path):
    holder = {}

    class CancelAfterOneTool(FakeChatModel):
        def invoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 2:
                holder["session"].cancel()
            if len(self.calls) <= 2:
                return TOOL_CALL
            return AIMessage(content="Réponse suivante")

    model = CancelAfterOneTool()
    session, _ = _session(tmp_path, model)
    holder["session"] = session

    session.handle("annulée après un outil")
    session.handle("suivante")

    sent = model.calls[2]
    assert [type(m).__name__ for m in sent[-4:]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "HumanMessage",
    ]


def test_cancel_tells_whether_it_took_effect(tmp_path):
    results = []

    class SpyModel(FakeChatModel):
        def invoke(self, messages):
            results.append(session.cancel())
            return super().invoke(messages)

    session, _ = _session(tmp_path, SpyModel())

    assert session.cancel() is False
    session.handle("bonjour")

    assert results == [True]
