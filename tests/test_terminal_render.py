from iacli.adapters.cli_chat import CliChatPresenter
from iacli.adapters.terminal_render import render_diff, render_markdown

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
