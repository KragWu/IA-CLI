import asyncio
import threading
import time

from langchain_core.messages import AIMessage

from iacli.adapters.textual_chat import ChatApp
from iacli.infrastructure.workspace_tools import create_workspace_tools
from iacli.services.chat import ChatSession
from tests.fakes import FakeChatModel

TOOL_CALL = AIMessage(
    content="", tool_calls=[{"name": "list_files", "args": {}, "id": "1"}]
)


def _app(tmp_path, model=None, history=None):
    def create_session(presenter):
        return ChatSession(
            model or FakeChatModel(),
            tmp_path,
            "",
            "fake-model",
            presenter=presenter,
            tools_factory=create_workspace_tools,
        )

    return ChatApp(create_session, history=history)


def _run(scenario):
    asyncio.run(scenario())


def _transcript(app):
    return "\n".join(app.transcript)


async def _settled(app):
    """Attend la fin du tour en cours (thread de travail de l'application)."""
    thread = app._turn_thread
    if thread is not None:
        await asyncio.to_thread(thread.join, 10)


async def _send(app, pilot, text):
    await pilot.click("#prompt")
    await pilot.press(*text, "enter")
    await _settled(app)
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
            assert app.query_one("#prompt").text == ""

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
            await _settled(app)
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
            await _settled(app)
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


def test_ansi_lines_are_displayed_as_styled_text_without_escape_codes(tmp_path):
    async def scenario():
        def create_session(presenter):
            return ChatSession(
                FakeChatModel(),
                tmp_path,
                "",
                "fake-model",
                presenter=presenter,
                tools_factory=create_workspace_tools,
            )

        app = ChatApp(create_session)
        async with app.run_test() as pilot:
            app._write(chr(27) + "[1mgras" + chr(27) + "[0m")
            await pilot.pause()
            log = app.query_one("#log")
            rendered = " ".join(strip.text for strip in log.lines)

            assert "gras" in rendered
            assert chr(27) not in rendered

    _run(scenario)


WRITE_CALL = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "write_file",
            "args": {"path": "a.txt", "content": "salut"},
            "id": "w1",
        }
    ],
)


async def _wait_for_confirmation(app, pilot):
    for _ in range(100):
        await pilot.pause(0.05)
        if app.query_one("#question").display:
            return
    raise AssertionError("aucune confirmation affichée")


async def _answer_write(tmp_path, *keys):
    model = FakeChatModel([WRITE_CALL, AIMessage(content="fini")])
    app = _app(tmp_path, model)
    async with app.run_test() as pilot:
        await pilot.click("#prompt")
        await pilot.press(*"ecris", "enter")
        await _wait_for_confirmation(app, pilot)
        assert "Modification proposée pour a.txt" in _transcript(app)
        await pilot.press(*keys)
        await _settled(app)
        await pilot.pause()
        assert not app.query_one("#question").display
        assert not app.query_one("#prompt").disabled
        return app


def test_a_file_write_is_applied_after_the_user_confirms(tmp_path):
    asyncio.run(_answer_write(tmp_path, "o"))

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "salut"


def test_a_file_write_is_refused_by_default(tmp_path):
    for keys in (("n",), ("enter",), ("escape",)):
        asyncio.run(_answer_write(tmp_path, *keys))

        assert not (tmp_path / "a.txt").exists()


def test_ctrl_c_during_a_confirmation_refuses_and_cancels_the_turn(tmp_path):
    app = asyncio.run(_answer_write(tmp_path, "ctrl+c"))

    assert not (tmp_path / "a.txt").exists()
    assert "Génération annulée" in app.transcript


def test_the_question_stays_displayed_until_a_valid_answer(tmp_path):
    async def scenario():
        model = FakeChatModel([WRITE_CALL, AIMessage(content="fini")])
        app = _app(tmp_path, model)
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press(*"ecris", "enter")
            await _wait_for_confirmation(app, pilot)
            await pilot.press("x", "a")

            assert app.query_one("#question").display
            await pilot.press("n")
            await _settled(app)

    asyncio.run(scenario())


def _text_of(app, selector):
    return str(app.query_one(selector).render())


def test_the_status_bar_shows_the_model_and_then_the_colored_context(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            assert "modèle fake-model" in _text_of(app, "#status")

            app.set_context((500, 1000))
            await pilot.pause()
            assert "contexte 500 / 1 000 tokens (50 %)" in _text_of(app, "#status")
            assert app.context_style == "green"

            app.set_context((700, 1000))
            assert app.context_style == "yellow"
            app.set_context((900, 1000))
            assert app.context_style == "red"

            app.set_context(None)
            await pilot.pause()
            assert "contexte" not in _text_of(app, "#status")

    _run(scenario)


def test_the_progress_indicator_is_shown_during_a_turn_and_hidden_after(tmp_path):
    release = threading.Event()

    class SlowModel(FakeChatModel):
        def invoke(self, messages):
            release.wait(5)
            return super().invoke(messages)

    async def scenario():
        app = _app(tmp_path, SlowModel())
        async with app.run_test() as pilot:
            assert not app.query_one("#progress").display

            await pilot.click("#prompt")
            await pilot.press(*"lent", "enter")
            await pilot.pause(0.3)

            assert app.query_one("#progress").display
            assert "L'agent traite votre demande" in _text_of(app, "#progress")

            release.set()
            await _settled(app)
            await pilot.pause()
            assert not app.query_one("#progress").display

    _run(scenario)


def test_the_progress_message_can_change_and_the_indicator_pauses(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            app.start_progress()
            app.set_progress_message("lecture de a.txt")
            await pilot.pause(0.3)
            assert "lecture de a.txt" in _text_of(app, "#progress")

            app.pause_progress()
            assert not app.query_one("#progress").display
            app.resume_progress()
            assert app.query_one("#progress").display

            app.stop_progress()
            app.resume_progress()
            assert not app.query_one("#progress").display

    _run(scenario)


def _prompt_text(app):
    return app.query_one("#prompt").text


def test_a_trailing_backslash_continues_on_the_next_line_instead_of_sending(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press("a", "backslash", "enter", "b")

            assert _prompt_text(app) == "a" + chr(10) + "b"
            assert not any(line.startswith("vous>") for line in app.transcript)

            await pilot.press("enter")
            await _settled(app)
            assert "vous> a" + chr(10) + "b" in app.transcript

    _run(scenario)


def test_ctrl_j_inserts_a_new_line(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press("a", "ctrl+j", "b")

            assert _prompt_text(app) == "a" + chr(10) + "b"

    _run(scenario)


def test_tab_completes_a_unique_slash_command(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press("slash", "h", "tab")

            assert _prompt_text(app) == "/help"

    _run(scenario)


def test_tab_completes_the_common_prefix_and_lists_the_candidates(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press("slash", "c", "tab")

            assert _prompt_text(app) == "/c"
            hint = str(app.query_one("#hint").render())
            assert "/clear" in hint
            assert "/context" in hint

            await pilot.press("l")
            assert not app.query_one("#hint").display

    _run(scenario)


def test_the_arrows_browse_the_history_and_restore_the_draft(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _send(app, pilot, "un")
            await _send(app, pilot, "deux")
            await pilot.press("t", "r")
            await pilot.press("up")
            assert _prompt_text(app) == "deux"
            await pilot.press("up")
            assert _prompt_text(app) == "un"
            await pilot.press("down", "down")
            assert _prompt_text(app) == "tr"

    _run(scenario)


def test_the_up_arrow_moves_in_a_multiline_message_before_browsing_the_history(
    tmp_path,
):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test() as pilot:
            await _send(app, pilot, "ancien")
            await pilot.press("a", "ctrl+j", "b", "up")

            assert _prompt_text(app) == "a" + chr(10) + "b"

    _run(scenario)


class _MemoryStore:
    def __init__(self, entries=()):
        self.entries = list(entries)

    def load(self):
        return list(self.entries)

    def append(self, text):
        self.entries.append(text)


def test_sent_messages_are_saved_and_the_saved_history_is_browsable(tmp_path):
    async def scenario():
        store = _MemoryStore(["ancien"])
        app = _app(tmp_path, history=store)
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press("up")
            assert _prompt_text(app) == "ancien"
            await pilot.press("down")

            await _send(app, pilot, "nouveau")

            assert store.entries == ["ancien", "nouveau"]

    _run(scenario)


def test_quitting_does_not_wait_for_a_model_call_in_progress(tmp_path):
    release = threading.Event()

    class BlockedModel(FakeChatModel):
        def invoke(self, messages):
            release.wait(30)
            return super().invoke(messages)

    async def scenario():
        app = _app(tmp_path, BlockedModel())
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press(*"bloque", "enter")
            await pilot.pause(0.2)
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert not app.is_running

    started = time.perf_counter()
    try:
        asyncio.run(scenario())
        elapsed = time.perf_counter() - started
    finally:
        release.set()

    assert elapsed < 5


def test_a_second_ctrl_c_during_a_cancelling_turn_quits(tmp_path):
    release = threading.Event()

    class BlockedModel(FakeChatModel):
        def invoke(self, messages):
            release.wait(30)
            return super().invoke(messages)

    async def scenario():
        app = _app(tmp_path, BlockedModel())
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press(*"bloque", "enter")
            await pilot.pause(0.2)
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.is_running

            await pilot.press("ctrl+c")
            await pilot.pause()
            assert not app.is_running

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_a_ctrl_c_pressed_before_the_turn_has_started_is_not_lost(tmp_path):
    release = threading.Event()

    class SlowModel(FakeChatModel):
        def invoke(self, messages):
            release.wait(5)
            return TOOL_CALL

    async def scenario():
        app = _app(tmp_path, SlowModel())
        real_handle = app._session.handle

        def late_handle(text):
            time.sleep(0.4)  # le thread de travail n'est pas encore entré dans `handle`
            return real_handle(text)

        app._session.handle = late_handle
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press(*"vite", "enter")
            await pilot.press("ctrl+c")
            await pilot.pause(0.8)
            release.set()
            await _settled(app)
            await pilot.pause()

            assert "Génération annulée" in app.transcript
            assert app.is_running

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_an_unexpected_failure_in_a_turn_shows_the_error_and_unlocks_the_prompt(
    tmp_path,
):
    async def scenario():
        app = _app(tmp_path)

        def failing(text):
            raise RuntimeError("panne inattendue")

        app._session.handle = failing
        async with app.run_test() as pilot:
            await _send(app, pilot, "bonjour")

            assert any(
                line.startswith("erreur>") and "panne inattendue" in line
                for line in app.transcript
            )
            assert not app.query_one("#prompt").disabled
            assert app.is_running

    _run(scenario)


def test_the_confirmation_accepts_capital_letters_too(tmp_path):
    asyncio.run(_answer_write(tmp_path, "O"))

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "salut"
