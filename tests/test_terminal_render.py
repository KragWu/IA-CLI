import os
from functools import partial
from io import StringIO

from markdown_it import MarkdownIt
from rich.console import Console
from rich.text import Text

from iacli.adapters.cli_chat import RESPONSE_MARGIN, CliChatPresenter
from iacli.adapters.terminal_render import (
    render_diff,
    render_markdown,
    supports_ansi,
)

BOLD = "\x1b[1m"


def test_markdown_emphasis_is_rendered_without_its_markers():
    rendered = render_markdown("Un mot **gras**")

    assert "**" not in rendered
    assert "gras" in rendered
    assert BOLD in rendered


def test_markdown_code_block_keeps_its_content_and_is_colored():
    rendered = render_markdown("```python\nprint('salut')\n```")

    assert "```" not in rendered
    assert "print" in rendered
    assert "salut" in rendered
    assert "\x1b[" in rendered


def test_markdown_headings_and_lists_lose_their_markers():
    rendered = render_markdown("# Titre\n\n- premier\n- second")

    assert "# Titre" not in rendered
    assert "Titre" in rendered
    assert "premier" in rendered
    assert "second" in rendered


def test_presenter_renders_markdown_responses_when_a_renderer_is_given():
    output = []
    presenter = CliChatPresenter(
        output_fn=output.append, render_markdown=render_markdown
    )

    presenter.show_response("Un mot **gras**")

    assert len(output) == 1
    assert output[0].startswith("\niacli> ")
    assert "**" not in output[0]
    assert "gras" in output[0]


def test_presenter_shows_the_raw_response_without_a_renderer():
    output = []
    presenter = CliChatPresenter(output_fn=output.append)

    presenter.show_response("Un mot **gras**")

    assert output == ["\niacli> Un mot **gras**\n"]


DIFF = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-ancien\n+nouveau\n"


def test_diff_keeps_every_line_and_is_colored():
    rendered = render_diff(DIFF)

    for text in ("a/f.py", "@@ -1 +1 @@", "ancien", "nouveau"):
        assert text in rendered
    assert "\x1b[" in rendered


def test_diff_colors_added_and_removed_lines_differently():
    lines = render_diff(DIFF).splitlines()
    removed = next(line for line in lines if "ancien" in line)
    added = next(line for line in lines if "nouveau" in line)

    assert removed.split("ancien")[0] != added.split("nouveau")[0]


def test_presenter_renders_the_diff_of_a_file_write_when_a_renderer_is_given():
    output = []
    presenter = CliChatPresenter(
        input_fn=lambda _prompt: "o",
        output_fn=output.append,
        render_diff=render_diff,
    )

    assert presenter.confirm_file_write("f.py", DIFF) is True

    shown = next(line for line in output if "nouveau" in line)
    assert "\x1b[" in shown


def test_presenter_shows_the_raw_diff_without_a_renderer():
    output = []
    presenter = CliChatPresenter(input_fn=lambda _prompt: "n", output_fn=output.append)

    presenter.confirm_file_write("f.py", DIFF)

    assert DIFF in output


def _plain(rendered):
    return Text.from_ansi(rendered).plain


def test_markdown_wraps_within_the_terminal_width_minus_the_margin(monkeypatch):
    monkeypatch.setattr(
        "iacli.adapters.terminal_render.shutil.get_terminal_size",
        lambda: os.terminal_size((40, 24)),
    )
    text = " ".join(["mot"] * 60)

    lines = _plain(render_markdown(text, margin=7)).splitlines()

    assert len(lines) > 1
    assert max(len(line) for line in lines) <= 40 - 7


def test_markdown_keeps_a_minimum_width_in_a_tiny_terminal(monkeypatch):
    monkeypatch.setattr(
        "iacli.adapters.terminal_render.shutil.get_terminal_size",
        lambda: os.terminal_size((5, 24)),
    )

    assert "bonjour" in _plain(render_markdown("bonjour", margin=7))


def test_the_following_lines_of_a_rendered_response_are_aligned_under_the_prefix():
    output = []
    presenter = CliChatPresenter(
        output_fn=output.append, render_markdown=lambda text: "un\ndeux\ntrois"
    )

    presenter.show_response("peu importe")

    assert output == ["\niacli> un\n       deux\n       trois\n"]


def test_the_raw_response_is_left_untouched_without_a_renderer():
    output = []
    presenter = CliChatPresenter(output_fn=output.append)

    presenter.show_response("un\ndeux")

    assert output == ["\niacli> un\ndeux\n"]


def test_ansi_output_needs_a_real_terminal():
    assert supports_ansi(Console(file=StringIO(), force_terminal=False)) is False
    assert (
        supports_ansi(
            Console(file=StringIO(), force_terminal=True, legacy_windows=False)
        )
        is True
    )


def test_ansi_output_is_refused_on_a_legacy_windows_console():
    console = Console(file=StringIO(), force_terminal=True, legacy_windows=True)

    assert supports_ansi(console) is False


def test_a_rendered_response_never_reaches_the_last_terminal_column(monkeypatch):
    monkeypatch.setattr(
        "iacli.adapters.terminal_render.shutil.get_terminal_size",
        lambda: os.terminal_size((40, 24)),
    )
    output = []
    presenter = CliChatPresenter(
        output_fn=output.append,
        render_markdown=partial(render_markdown, margin=RESPONSE_MARGIN),
    )
    code = (
        "```python"
        + chr(10)
        + "print('une ligne de code assez longue')"
        + chr(10)
        + "```"
    )

    presenter.show_response(
        "Voici :"
        + chr(10)
        + chr(10)
        + code
        + chr(10)
        + chr(10)
        + " ".join(["mot"] * 30)
    )

    lines = _plain(output[0]).splitlines()
    assert len(lines) > 3
    assert max(len(line) for line in lines) < 40


def test_angle_brackets_in_prose_are_kept_as_text():
    plain = _plain(render_markdown("Utilise List<String> et <tool_call> ici"))

    assert "List<String>" in plain
    assert "<tool_call>" in plain


def test_html_looking_blocks_are_shown_not_swallowed():
    plain = _plain(render_markdown("<div>bloc html</div>"))

    assert "<div>bloc html</div>" in plain


def test_markdown_formatting_still_works_next_to_angle_brackets():
    rendered = render_markdown("**gras** et List<String>")

    assert "**" not in rendered
    assert "List<String>" in _plain(rendered)


def test_ansi_output_is_refused_on_a_dumb_terminal(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    console = Console(file=StringIO(), force_terminal=True, legacy_windows=False)

    assert supports_ansi(console) is False


def test_the_response_is_parsed_only_once(monkeypatch):
    parsed = []
    real_parse = MarkdownIt.parse

    def spy(self, src, env=None):
        parsed.append(src)
        return real_parse(self, src, env)

    monkeypatch.setattr(MarkdownIt, "parse", spy)

    render_markdown("**gras** et List<String>")

    assert parsed.count("**gras** et List<String>") == 1
