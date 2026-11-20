import pytest

from iacli.domain.commands import (
    COMMANDS,
    common_prefix,
    matching_commands,
    resolve_enter,
)


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


def test_commands_matching_a_typed_prefix_ignore_the_case():
    assert [c.name for c in matching_commands("/C", COMMANDS)] == [
        "/clear",
        "/context",
    ]


@pytest.mark.parametrize("text", ["help", "/help me", "/zzz", ""])
def test_nothing_matches_outside_a_lone_slash_word(text):
    assert matching_commands(text, COMMANDS) == []


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["/clear", "/context"], "/c"),
        (["/help"], "/help"),
        (["/exit", "/quit"], "/"),
        (["/ab", "/abc"], "/ab"),
        ([], ""),
    ],
)
def test_the_common_prefix_of_several_names(names, expected):
    assert common_prefix(names) == expected
