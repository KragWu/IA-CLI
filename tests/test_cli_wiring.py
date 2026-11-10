import io
import sys

from iacli import cli
from iacli.adapters.cli_chat import PromptStatus
from iacli.domain.commands import COMMANDS


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
    monkeypatch.setattr(cli, "create_prompt_input", lambda *args, **kwargs: input)
    monkeypatch.setattr(sys, "stdin", _Tty())
    rich = cli.create_chat_presenter(tmp_path / "history")

    assert plain._render_markdown is None
    assert rich._render_markdown is not None
