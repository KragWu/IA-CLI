import pytest

from iacli.domain.text import neutralize_control_characters


def test_plain_text_is_left_untouched():
    text = "Bonjour é à 你好 😀\n\tindenté"

    assert neutralize_control_characters(text) == text


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("\x1b]0;titre", "\\x1b]0;titre"),
        ("a\rb", "a\\x0db"),
        ("\x07", "\\x07"),
        ("\x7f", "\\x7f"),
        ("\x9b31m", "\\x9b31m"),
        ("\x00", "\\x00"),
    ],
)
def test_control_characters_become_visible_escapes(raw, shown):
    assert neutralize_control_characters(raw) == shown


def test_a_crlf_line_ending_is_shown_as_a_plain_line_break():
    assert neutralize_control_characters("a\r\nb\r\n") == "a\nb\n"


def test_a_lone_carriage_return_is_still_made_visible():
    assert neutralize_control_characters("a\r\r\nb\rc") == "a\\x0d\nb\\x0dc"
