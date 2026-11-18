import threading

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
    snippet = "    def f():\n        return 1\n"

    session.handle(snippet)

    (sent,) = [m for m in model.calls[0] if isinstance(m, HumanMessage)]
    assert sent.content == snippet


def test_a_command_is_still_recognised_around_whitespace(tmp_path):
    session, output = _session(tmp_path)

    assert session.handle("  /exit  ") is False
    assert session.handle("\n/help\n") is True
    assert "Commandes disponibles :" in output
