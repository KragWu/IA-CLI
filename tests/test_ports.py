from iacli.services.ports import ChatInput, ChatPresenter, ConsolePresenter


def test_the_presenter_port_of_a_session_does_not_read_the_input():
    assert "prompt" not in vars(ChatPresenter)


def test_the_console_loop_needs_a_presenter_that_also_reads_the_input():
    assert "prompt" in vars(ChatInput)
    assert {ChatPresenter, ChatInput} <= set(ConsolePresenter.__mro__)


def test_the_file_history_store_satisfies_the_history_port(tmp_path):
    from iacli.infrastructure.history import FileHistoryStore
    from iacli.services.ports import HistoryStore

    store: HistoryStore = FileHistoryStore(tmp_path / "history")

    assert store.load() == []


def test_the_chat_session_satisfies_the_chat_turns_port(tmp_path):
    from iacli.infrastructure.workspace_tools import create_workspace_tools
    from iacli.services.chat import ChatSession
    from iacli.services.ports import ChatTurns
    from tests.fakes import FakeChatModel

    class Presenter:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    session: ChatTurns = ChatSession(
        FakeChatModel(),
        tmp_path,
        "",
        "fake-model",
        presenter=Presenter(),
        tools_factory=create_workspace_tools,
    )

    assert session.cancel() is False
