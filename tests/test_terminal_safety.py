from pathlib import Path

from iacli.adapters.cli_chat import CliChatPresenter

ESC = chr(27)
CR = chr(13)


def _presenter(**options):
    output = []
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: "o", output_fn=output.append, **options
    )
    return presenter, output


def _assert_safe(output):
    joined = "".join(output)
    assert ESC not in joined
    assert CR not in joined


def test_a_response_cannot_send_control_characters_to_the_terminal():
    presenter, output = _presenter()

    presenter.show_response(f"titre {ESC}]0;PIRATE\nligne{CR}écrasée")

    _assert_safe(output)
    assert "\\x1b]0;PIRATE" in "".join(output)


def test_a_response_is_neutralized_before_the_markdown_rendering():
    seen = []
    presenter, _ = _presenter(render_markdown=lambda text: seen.append(text) or text)

    presenter.show_response(f"a{ESC}b")

    assert seen == ["a\\x1bb"]


def test_an_error_message_cannot_send_control_characters():
    presenter, output = _presenter()

    presenter.show_error(f"échec {ESC}[2J")

    _assert_safe(output)


def test_a_proposed_diff_is_neutralized_before_the_diff_rendering():
    seen = []
    presenter, output = _presenter(render_diff=lambda text: seen.append(text) or text)

    presenter.confirm_file_write("a.txt", f"+{ESC}]52;c;x")

    assert seen == ["+\\x1b]52;c;x"]
    _assert_safe(output)


def test_a_proposed_diff_and_path_are_neutralized_without_rendering():
    presenter, output = _presenter()

    presenter.confirm_file_write(f"a{ESC}.txt", f"+{ESC}[2J")

    _assert_safe(output)


def test_a_proposed_shell_command_cannot_rewrite_the_screen():
    presenter, output = _presenter()

    presenter.confirm_shell_command(f"rm -rf x{CR}echo inoffensif", Path("."))

    _assert_safe(output)
    assert "rm -rf x\\x0decho inoffensif" in "".join(output)


def _broken(_text):
    raise RuntimeError("rendu cassé")


def test_a_response_is_shown_as_plain_text_when_the_rendering_fails():
    presenter, output = _presenter(render_markdown=_broken)

    presenter.show_response("réponse du modèle")

    assert "iacli> réponse du modèle" in "".join(output)


def test_a_diff_is_shown_as_plain_text_when_the_rendering_fails():
    presenter, output = _presenter(render_diff=_broken)

    assert presenter.confirm_file_write("a.txt", "+ligne") is True

    assert "+ligne" in output
