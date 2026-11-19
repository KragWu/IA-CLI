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


@pytest.mark.parametrize(
    "code",
    [
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
        0x200E,
        0x200F,
        0x061C,
        0x2028,
        0x2029,
        0x200B,
        0x2060,
        0xFEFF,
    ],
)
def test_invisible_and_reordering_characters_become_visible_escapes(code):
    assert neutralize_control_characters(f"a{chr(code)}b") == f"a\\u{code:04x}b"


@pytest.mark.parametrize(
    "code",
    [0x00AD, 0x034F, 0x2061, 0x2064, 0x115F, 0x1160, 0x3164, 0xFFA0, 0x180E],
)
def test_other_invisible_characters_become_visible_escapes(code):
    assert neutralize_control_characters(f"a{chr(code)}b") == f"a\\u{code:04x}b"


def test_unicode_tag_characters_hiding_text_become_visible_escapes():
    hidden = "".join(chr(0xE0000 + ord(char)) for char in "rm")

    assert neutralize_control_characters(f"ls{hidden}") == "ls\\U000e0072\\U000e006d"


@pytest.mark.parametrize("text", ["👨‍👩‍👧", "می‌خواهم", "é́"])
def test_joiners_and_combining_marks_are_kept(text):
    assert neutralize_control_characters(text) == text


def test_a_lone_carriage_return_is_still_made_visible():
    assert neutralize_control_characters("a\r\r\nb\rc") == "a\\x0d\nb\\x0dc"
