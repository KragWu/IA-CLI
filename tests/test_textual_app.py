import asyncio
import threading

from langchain_core.messages import AIMessage

from iacli.adapters.textual_chat import ChatApp
from iacli.infrastructure.workspace_tools import create_workspace_tools
from iacli.services.chat import ChatSession
from tests.fakes import FakeChatModel

TOOL_CALL = AIMessage(
    content="", tool_calls=[{"name": "list_files", "args": {}, "id": "1"}]
)


def _app(tmp_path, model=None):
    def create_session(presenter):
        return ChatSession(
            model or FakeChatModel(),
            tmp_path,
            "",
            "fake-model",
            presenter=presenter,
            tools_factory=create_workspace_tools,
        )

    return ChatApp(create_session)


def _run(scenario):
    asyncio.run(scenario())


def _transcript(app):
    return "\n".join(app.transcript)


async def _send(app, pilot, text):
    await pilot.click("#prompt")
    await pilot.press(*text, "enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


def test_the_app_announces_the_model_when_it_starts(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test():
            assert "fake-model" in _transcript(app)

    _run(scenario)


def test_a_sent_message_is_echoed_and_answered(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _send(app, pilot, "bonjour")

            assert "vous> bonjour" in app.transcript
            assert "iacli> Réponse 1" in _transcript(app)

    _run(scenario)


def test_the_input_is_cleared_and_usable_again_after_the_answer(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _send(app, pilot, "un")
            await _send(app, pilot, "deux")

            assert "iacli> Réponse 2" in _transcript(app)
            assert app.query_one("#prompt").value == ""

    _run(scenario)


def test_the_exit_command_closes_the_app(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _send(app, pilot, "/exit")

            assert app.return_code is not None or not app.is_running

    _run(scenario)


def test_the_interface_stays_responsive_while_the_model_is_working(tmp_path):
    release = threading.Event()

    class SlowModel(FakeChatModel):
        def invoke(self, messages):
            release.wait(5)
            return super().invoke(messages)

    async def scenario():
        app = _app(tmp_path, SlowModel())
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press(*"lent", "enter")
            await pilot.pause()

            assert "vous> lent" in app.transcript
            assert app.query_one("#prompt").disabled

            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app.query_one("#prompt").disabled

    _run(scenario)


def test_ctrl_c_during_a_turn_cancels_it_without_leaving_the_app(tmp_path):
    release = threading.Event()

    class SlowModel(FakeChatModel):
        def invoke(self, messages):
            self.calls.append(messages)
            release.wait(5)
            return TOOL_CALL

    async def scenario():
        app = _app(tmp_path, SlowModel())
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press(*"longue", "enter")
            await pilot.pause()
            await pilot.press("ctrl+c")
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert "Génération annulée" in app.transcript
            assert app.is_running

    _run(scenario)


def test_ctrl_c_and_ctrl_d_at_the_prompt_quit_immediately(tmp_path):
    async def scenario(key):
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.press(key)
            await pilot.pause()

            assert not app.is_running

    for key in ("ctrl+c", "ctrl+d"):
        asyncio.run(scenario(key))
