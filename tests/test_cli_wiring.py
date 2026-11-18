import io
import sys

from iacli import cli
from iacli.adapters.cli_chat import RESPONSE_MARGIN
from iacli.domain.commands import COMMANDS
from iacli.domain.status import PromptStatus


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_presenter_uses_the_plain_input_outside_a_terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "stdin", io.StringIO("ligne lue\n"))

    presenter = cli.create_chat_presenter(tmp_path / "history")

    assert presenter.prompt() == "ligne lue"


def test_presenter_uses_the_rich_prompt_in_an_interactive_terminal(
    monkeypatch, tmp_path
):
    created = {}

    def fake_create_prompt_input(commands, history_path, status):
        created.update(commands=commands, history_path=history_path, status=status)
        return lambda prompt: "saisie riche"

    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(cli, "create_prompt_input", fake_create_prompt_input)

    presenter = cli.create_chat_presenter(tmp_path / "history")

    assert presenter.prompt() == "saisie riche"
    assert created["commands"] == COMMANDS
    assert created["history_path"] == tmp_path / "history"
    presenter.show_context_usage(1, 2)
    assert created["status"]() == PromptStatus("", 1, 2)


def test_presenter_renders_markdown_only_in_an_interactive_terminal(
    monkeypatch, tmp_path
):
    plain = cli.create_chat_presenter(tmp_path / "history")
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(cli, "supports_ansi", lambda: True)
    monkeypatch.setattr(cli, "create_prompt_input", lambda *args, **kwargs: input)
    monkeypatch.setattr(sys, "stdin", _Tty())
    rich = cli.create_chat_presenter(tmp_path / "history")

    assert plain._render_markdown is None
    assert rich._render_markdown is not None
    assert rich._render_diff is not None


def test_presenter_stays_plain_on_a_terminal_without_ansi_support(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(cli, "supports_ansi", lambda: False)

    presenter = cli.create_chat_presenter(tmp_path / "history")

    assert presenter._render_markdown is None
    assert presenter._render_diff is None


def test_the_markdown_renderer_leaves_room_for_the_response_prefix(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(cli, "supports_ansi", lambda: True)

    presenter = cli.create_chat_presenter(tmp_path / "history")

    assert presenter._render_markdown.keywords == {"margin": RESPONSE_MARGIN}
