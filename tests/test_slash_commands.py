import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.domain.commands import (
    COMMANDS,
    command_names,
    find_command,
    is_command_input,
)
from iacli.infrastructure.workspace_tools import create_workspace_tools
from iacli.services.chat import run_chat
from tests.fakes import FakeChatModel


def _run(tmp_path, prompts, model=None, context_limit=None):
    iterator = iter(prompts)
    output = []
    model = model or FakeChatModel()
    run_chat(
        model,
        tmp_path,
        "",
        "fake-model",
        presenter=CliChatPresenter(
            input_fn=lambda _prompt: next(iterator), output_fn=output.append
        ),
        tools_factory=create_workspace_tools,
        context_limit=context_limit,
    )
    return output, model


def test_registry_lists_the_available_commands():
    assert command_names() == ["/help", "/clear", "/context", "/exit", "/quit"]
    assert all(command.description for command in COMMANDS)


def test_find_command_ignores_case_and_surrounding_spaces():
    assert find_command("  /HELP ") is not None
    assert find_command("/inconnue") is None
    assert find_command("bonjour") is None


def test_help_lists_every_command_with_its_description(tmp_path):
    output, model = _run(tmp_path, ["/help", "/exit"])

    shown = "\n".join(output)
    for command in COMMANDS:
        assert command.name in shown
        assert command.description in shown
    assert model.calls == []


def test_clear_forgets_the_conversation_but_keeps_the_instructions(tmp_path):
    output, model = _run(tmp_path, ["premier", "/clear", "second", "/exit"])

    assert "Historique effacé" in output
    second_call = model.calls[1]
    assert isinstance(second_call[0], SystemMessage)
    assert [m.content for m in second_call if isinstance(m, HumanMessage)] == ["second"]


def test_context_reports_the_usage_of_the_last_turn(tmp_path):
    usage = {"input_tokens": 400, "output_tokens": 100, "total_tokens": 500}
    model = FakeChatModel([AIMessage(content="ok", usage_metadata=usage)])

    output, _ = _run(tmp_path, ["bonjour", "/context", "/exit"], model, 1000)

    assert sum(line.startswith("contexte> 500 / 1 000") for line in output) == 2


def test_context_without_measurement_says_so(tmp_path):
    output, _ = _run(tmp_path, ["/context", "/exit"], context_limit=1000)

    assert any("aucune mesure" in line for line in output)


def test_unknown_command_is_an_error_and_is_not_sent_to_the_model(tmp_path):
    output, model = _run(tmp_path, ["/inconnue", "/exit"])

    assert any(
        line.startswith("erreur>") and "/inconnue" in line and "/help" in line
        for line in output
    )
    assert model.calls == []


def test_a_message_containing_a_path_is_not_taken_for_a_command(tmp_path):
    _, model = _run(tmp_path, ["/etc/hosts est lisible ?", "/exit"])

    assert len(model.calls) == 1


def test_an_unknown_word_followed_by_text_is_a_message_for_the_model(tmp_path):
    output, model = _run(tmp_path, ["/tmp contient quoi ?", "/exit"])

    assert len(model.calls) == 1
    assert not any(line.startswith("erreur>") for line in output)


def test_a_known_command_accepts_trailing_text(tmp_path):
    output, model = _run(tmp_path, ["/help merci", "/exit"])

    assert "Commandes disponibles :" in output
    assert model.calls == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/help", True),
        ("/HELP encore", True),
        ("/inconnue", True),
        ("  /inconnue  ", True),
        ("/tmp contient quoi ?", False),
        ("/etc/hosts", False),
        ("bonjour", False),
        ("", False),
    ],
)
def test_is_command_input_needs_a_known_command_or_a_lone_word(text, expected):
    assert is_command_input(text) is expected
