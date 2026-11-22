import pytest

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.adapters.terminal_input import render_toolbar
from iacli.domain.status import (
    PromptStatus,
    context_level,
    context_percent,
    format_duration,
    format_tokens,
)


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


def test_duration_formatter_uses_milliseconds_seconds_minutes_and_hours():
    assert format_duration(0.5) == "500 ms"
    assert format_duration(1.25) == "1.25 s"
    assert format_duration(60) == "1 min 00 s"
    assert format_duration(251) == "4 min 11 s"
    assert format_duration(3600) == "1 h 00 min 00 s"
    assert format_duration(3661) == "1 h 01 min 01 s"


@pytest.mark.parametrize(
    ("used", "expected"),
    [
        (0, "ok"),
        (59, "ok"),
        (60, "warn"),
        (79, "warn"),
        (80, "danger"),
        (120, "danger"),
    ],
)
def test_the_context_level_turns_to_warn_at_60_percent_and_danger_at_80(used, expected):
    assert context_level(used, 100) == expected


@pytest.mark.parametrize(
    ("used", "limit", "expected"),
    [
        (500, 1000, 50),
        (1, 3, 33),
        (2, 3, 67),
        (1200, 1000, 120),
        (0, 1000, 0),
        (5, 0, 0),
    ],
)
def test_the_context_percent_is_rounded_and_tolerates_a_zero_limit(
    used, limit, expected
):
    assert context_percent(used, limit) == expected


def test_the_context_level_tolerates_a_zero_limit():
    assert context_level(5, 0) == "ok"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.9996, "1.00 s"),
        (59.996, "1 min 00 s"),
        (3599.7, "1 h 00 min 00 s"),
        (0.9994, "999 ms"),
        (59.99, "59.99 s"),
        (3599.4, "59 min 59 s"),
    ],
)
def test_a_duration_never_shows_a_rounded_value_out_of_its_range(seconds, expected):
    assert format_duration(seconds) == expected
