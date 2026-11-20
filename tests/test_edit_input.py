import pytest

from iacli.domain.commands import resolve_enter


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
