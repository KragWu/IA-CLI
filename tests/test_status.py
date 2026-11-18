import pytest

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.adapters.terminal_input import render_toolbar
from iacli.domain.status import PromptStatus, format_tokens


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (4500, "4 500"),
        (18000, "18 000"),
        (1234567, "1 234 567"),
    ],
)
def test_format_tokens_groups_thousands_with_spaces(count, expected):
    assert format_tokens(count) == expected


def test_the_status_bar_and_the_context_line_format_tokens_the_same_way():
    lines = []
    CliChatPresenter(output_fn=lines.append).show_context_usage(1234567, 2000000)

    toolbar = "".join(
        text for _style, text in render_toolbar(PromptStatus("m", 1234567, 2000000))
    )

    for shown in (lines[0], toolbar):
        assert format_tokens(1234567) in shown
        assert format_tokens(2000000) in shown
