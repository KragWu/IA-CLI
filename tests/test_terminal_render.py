from iacli.adapters.cli_chat import CliChatPresenter
from iacli.adapters.terminal_render import render_markdown

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
