from pathlib import Path

from iacli.adapters.textual_chat import TextualPresenter
from iacli.domain.commands import COMMANDS

ESC = chr(27)


def _presenter():
    lines = []
    return TextualPresenter(emit=lines.append), lines


def test_start_announces_the_model_and_the_project():
    presenter, lines = _presenter()

    presenter.start(Path("projet"), "qwen")

    assert "qwen" in lines[0]
    assert "projet" in lines[0]


def test_a_response_is_shown_after_the_response_prefix():
    presenter, lines = _presenter()

    presenter.show_response("Bonjour")

    assert any("iacli> Bonjour" in line for line in lines)


def test_everything_shown_is_neutralized():
    presenter, lines = _presenter()

    presenter.show_response(f"a{ESC}b")
    presenter.show_error(f"c{ESC}d")
    presenter.show_notice(f"e{ESC}f")
    presenter.show_tool_call(f"g{ESC}h", {"x": f"i{ESC}j"})
    presenter.show_tool_result(f"k{ESC}l")
    presenter.show_action_timing(f"m{ESC}n", 1.0)

    assert ESC not in "".join(lines)


def test_errors_notices_and_cancellation_are_distinguishable():
    presenter, lines = _presenter()

    presenter.show_error("panne")
    presenter.show_notice("info")
    presenter.show_cancelled()

    assert lines == ["erreur> panne", "info", "Génération annulée"]


def test_the_help_lists_every_command():
    presenter, lines = _presenter()

    presenter.show_help(COMMANDS)

    text = "\n".join(lines)
    assert all(command.name in text for command in COMMANDS)


def test_tool_progress_and_timing_lines_follow_the_console_wording():
    presenter, lines = _presenter()

    presenter.show_tool_call("read_file", {"path": "a.py"})
    presenter.show_tool_result("read_file")
    presenter.show_action_timing("read_file", 0.5)
    presenter.show_total_duration(1.25)

    assert lines == [
        'outil> read_file({"path": "a.py"})',
        "outil terminé> read_file",
        "temps action> read_file : 500 ms",
        "temps total> demande : 1.25 s",
    ]


def test_the_context_usage_is_shown_and_remembered_for_the_status_bar():
    presenter, lines = _presenter()

    presenter.show_context_usage(500, 1000)

    assert lines == ["contexte> 500 / 1 000 tokens (50 %)"]
    assert presenter.context_usage == (500, 1000)
    presenter.clear_context_usage()
    assert presenter.context_usage is None


def test_the_context_warning_mentions_the_share_used():
    presenter, lines = _presenter()

    presenter.show_context_warning(800, 1000)

    assert "80 %" in lines[0]


def test_confirmations_are_refused_until_the_modal_dialogs_exist():
    presenter, lines = _presenter()

    assert presenter.confirm_file_write("a.txt", "+x") is False
    assert presenter.confirm_shell_command("ls", Path(".")) is False
    assert len(lines) == 2


def test_a_response_goes_through_the_markdown_renderer_when_there_is_one():
    lines = []
    presenter = TextualPresenter(emit=lines.append, render_markdown=str.upper)

    presenter.show_response("**gras**")

    assert any("iacli> **GRAS**" in line for line in lines)


def test_the_text_is_neutralized_before_the_markdown_rendering():
    seen = []
    lines = []

    def render(text):
        seen.append(text)
        return text

    presenter = TextualPresenter(emit=lines.append, render_markdown=render)

    presenter.show_response(f"a{ESC}b")

    assert seen == ["a\\x1bb"]


def test_a_failing_markdown_renderer_falls_back_to_the_plain_text():
    lines = []

    def failing(text):
        raise ValueError("boom")

    presenter = TextualPresenter(emit=lines.append, render_markdown=failing)

    presenter.show_response("Bonjour")

    assert any("iacli> Bonjour" in line for line in lines)
