import threading

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.adapters.terminal_input import (
    SlashCommandCompleter,
    SlashCommandLexer,
    create_prompt_input,
    render_toolbar,
    resolve_enter,
)
from iacli.domain.commands import COMMANDS
from iacli.domain.status import PromptStatus


def _completions(text):
    document = Document(text, len(text))
    completer = SlashCommandCompleter(COMMANDS)
    return list(completer.get_completions(document, CompleteEvent()))


def test_slash_alone_proposes_every_registered_command():
    assert [c.text for c in _completions("/")] == [command.name for command in COMMANDS]


def test_completion_narrows_down_to_the_matching_commands():
    assert [c.text for c in _completions("/c")] == ["/clear", "/context"]


def test_completion_replaces_what_was_typed_and_describes_the_command():
    (completion,) = _completions("/he")

    assert completion.text == "/help"
    assert completion.start_position == -3
    assert "commandes" in completion.display_meta_text


@pytest.mark.parametrize("text", ["", "bonjour", "/help ", "voir /he", "a\n/he"])
def test_no_completion_outside_the_command_word(text):
    assert _completions(text) == []


def _first_line_styles(text):
    lexer = SlashCommandLexer()
    return lexer.lex_document(Document(text))(0)


def test_known_command_is_highlighted():
    assert _first_line_styles("/help")[0] == ("class:command", "/help")


def test_unknown_command_is_highlighted_as_an_error():
    assert _first_line_styles("/inconnue")[0] == ("class:unknown-command", "/inconnue")


def test_an_unknown_word_followed_by_text_is_not_flagged_as_a_command():
    assert _first_line_styles("/tmp contient quoi") == [("", "/tmp contient quoi")]


def test_a_command_followed_by_text_is_highlighted_as_an_error():
    styles = _first_line_styles("/help suite")

    assert styles == [("class:unknown-command", "/help"), ("", " suite")]


def test_the_style_follows_the_whole_text_not_only_its_first_line():
    lexer = SlashCommandLexer()

    plain = lexer.lex_document(Document("/tmp" + chr(10) + "suite"))(0)
    refused = lexer.lex_document(Document("/clear" + chr(10) + "suite"))(0)

    assert plain == [("", "/tmp")]
    assert refused == [("class:unknown-command", "/clear")]


def test_plain_text_keeps_the_default_style():
    assert _first_line_styles("bonjour") == [("", "bonjour")]


def _read(tmp_path, keys, then=""):
    """Lit une saisie simulée ; `then` est envoyé un peu plus tard, car la complétion
    de prompt_toolkit est asynchrone et doit être appliquée avant la touche suivante."""
    with create_pipe_input() as pipe:
        read = create_prompt_input(
            COMMANDS,
            history_path=tmp_path / "history",
            status=lambda: PromptStatus("m", None, None),
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text(keys)
        if then:
            threading.Timer(0.5, pipe.send_text, args=(then,)).start()
        return read("vous> ")


def test_enter_submits_the_message(tmp_path):
    assert _read(tmp_path, "bonjour\r") == "bonjour"


def test_tab_completes_a_slash_command(tmp_path):
    assert _read(tmp_path, "/he\t", then="\r") == "/help"


def test_backslash_then_enter_continues_on_a_new_line(tmp_path):
    assert _read(tmp_path, "ligne un\\\rligne deux\r") == "ligne un\nligne deux"


BACKSLASH = chr(92)
ENTER = chr(13)


def test_a_doubled_backslash_at_the_end_is_sent_as_a_single_one(tmp_path):
    keys = "dir C:" + BACKSLASH + "foo" + BACKSLASH * 2 + ENTER

    assert _read(tmp_path, keys) == "dir C:" + BACKSLASH + "foo" + BACKSLASH


def test_three_backslashes_keep_one_and_continue_on_a_new_line(tmp_path):
    keys = "p" + BACKSLASH * 3 + ENTER + "x" + ENTER

    assert _read(tmp_path, keys) == "p" + BACKSLASH + chr(10) + "x"


def test_a_backslash_inside_a_line_is_left_alone(tmp_path):
    keys = "a" + BACKSLASH + "b" + ENTER

    assert _read(tmp_path, keys) == "a" + BACKSLASH + "b"


def test_ctrl_j_inserts_a_new_line(tmp_path):
    assert _read(tmp_path, "a\x0ab\r") == "a\nb"


def test_ctrl_d_on_an_empty_line_ends_the_input(tmp_path):
    with pytest.raises(EOFError):
        _read(tmp_path, "\x04")


def test_ctrl_c_interrupts_the_input(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        _read(tmp_path, "texte\x03")


def test_submitted_messages_are_kept_in_the_history_file(tmp_path):
    _read(tmp_path, "mon message\r")

    assert "mon message" in (tmp_path / "history").read_text(encoding="utf-8")


def _toolbar_text(status):
    return "".join(text for _style, text in render_toolbar(status))


def test_toolbar_shows_the_model_alone_before_any_measure():
    assert _toolbar_text(PromptStatus("qwen", None, None)) == " modèle qwen "


def test_toolbar_shows_the_context_usage():
    text = _toolbar_text(PromptStatus("qwen", 4500, 18000))

    assert "qwen" in text
    assert "4 500 / 18 000 tokens (25 %)" in text


@pytest.mark.parametrize(
    ("used", "style"),
    [(1000, "class:ok"), (12000, "class:warn"), (16000, "class:danger")],
)
def test_toolbar_colors_the_usage_by_level(used, style):
    styles = [
        s for s, text in render_toolbar(PromptStatus("m", used, 18000)) if "%" in text
    ]

    assert styles == [style]


def test_presenter_status_follows_the_model_and_the_context(tmp_path):
    presenter = CliChatPresenter(output_fn=lambda _line: None)

    presenter.start(tmp_path, "qwen")
    presenter.show_context_usage(500, 1000)

    assert presenter.prompt_status() == PromptStatus("qwen", 500, 1000)


def test_presenter_reads_the_prompt_through_prompt_fn_but_confirms_with_input_fn():
    prompts = []
    presenter = CliChatPresenter(
        input_fn=lambda text: "o",
        prompt_fn=lambda text: prompts.append(text) or "message",
        output_fn=lambda _line: None,
    )

    assert presenter.prompt() == "message"
    assert (
        presenter.confirm_shell_command("ls", __import__("pathlib").Path(".")) is True
    )
    assert prompts == ["vous> "]


@pytest.mark.parametrize(
    ("backslashes", "kept", "submit"),
    [(0, 0, True), (1, 0, False), (2, 1, True), (3, 1, False), (4, 2, True)],
)
def test_resolve_enter_pairs_the_trailing_backslashes(backslashes, kept, submit):
    backslash = chr(92)

    text, sent = resolve_enter("abc" + backslash * backslashes)

    assert sent is submit
    expected = "abc" + backslash * kept + ("" if submit else chr(10))
    assert text == expected
