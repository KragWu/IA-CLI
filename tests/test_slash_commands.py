import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.domain.commands import COMMANDS, InputKind, parse_input
from iacli.infrastructure.workspace_tools import create_workspace_tools
from iacli.services.chat import ChatSession, run_chat
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
    assert [command.name for command in COMMANDS] == [
        "/help",
        "/clear",
        "/context",
        "/exit",
        "/quit",
    ]
    assert all(command.description for command in COMMANDS)


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


def test_clear_also_resets_the_context_shown_in_the_status_bar(tmp_path):
    usage = {"input_tokens": 400, "output_tokens": 100, "total_tokens": 500}
    model = FakeChatModel([AIMessage(content="ok", usage_metadata=usage)])
    prompts = iter(["bonjour", "/clear", "/exit"])
    statuses = []

    class RecordingPresenter(CliChatPresenter):
        def prompt(self):
            statuses.append(self.prompt_status())
            return super().prompt()

    presenter = RecordingPresenter(
        input_fn=lambda _prompt: next(prompts), output_fn=lambda _line: None
    )
    run_chat(
        model,
        tmp_path,
        "",
        "fake-model",
        presenter=presenter,
        tools_factory=create_workspace_tools,
        context_limit=1000,
    )

    # Invites successives : avant le tour, après le tour, après /clear.
    assert [status.context_used for status in statuses] == [None, 500, None]


@pytest.mark.parametrize(
    ("text", "kind", "word"),
    [
        ("/help", InputKind.COMMAND, "/help"),
        ("/HELP", InputKind.COMMAND, "/HELP"),
        ("  /exit  ", InputKind.COMMAND, "/exit"),
        ("/help merci", InputKind.COMMAND_WITH_ARGUMENTS, "/help"),
        ("/clear\nune question", InputKind.COMMAND_WITH_ARGUMENTS, "/clear"),
        ("/inconnue", InputKind.UNKNOWN_COMMAND, "/inconnue"),
        ("/tmp contient quoi ?", InputKind.MESSAGE, ""),
        ("/tmp\nsuite", InputKind.MESSAGE, ""),
        ("/etc/hosts", InputKind.MESSAGE, ""),
        ("bonjour", InputKind.MESSAGE, ""),
        ("", InputKind.MESSAGE, ""),
    ],
)
def test_parse_input_tells_commands_from_messages(text, kind, word):
    parsed = parse_input(text)

    assert parsed.kind is kind
    assert parsed.word == word


def test_parse_input_resolves_the_registered_command_ignoring_case():
    assert parse_input("  /HELP ").command is COMMANDS[0]
    assert parse_input("/inconnue").command is None
    assert parse_input("bonjour").command is None


def test_a_known_command_followed_by_text_is_refused_and_does_nothing(tmp_path):
    output, model = _run(
        tmp_path, ["premier", "/clear et une question", "second", "/exit"]
    )

    assert any(
        line.startswith("erreur>") and "/clear" in line and "argument" in line
        for line in output
    )
    assert "Historique effacé" not in output
    # L'historique est intact : le modèle revoit le premier message au second tour.
    assert [m.content for m in model.calls[1] if isinstance(m, HumanMessage)] == [
        "premier",
        "second",
    ]


def test_an_exit_command_with_text_does_not_end_the_session(tmp_path):
    output, model = _run(tmp_path, ["/exit bientôt", "bonjour", "/exit"])

    assert any(line.startswith("erreur>") and "/exit" in line for line in output)
    assert len(model.calls) == 1


def test_a_multiline_message_starting_with_a_command_word_is_not_lost(tmp_path):
    output, model = _run(tmp_path, ["/clear\nla vraie question", "/exit"])

    assert any(line.startswith("erreur>") for line in output)
    assert model.calls == []


USAGE = {"input_tokens": 400, "output_tokens": 100, "total_tokens": 500}


def _statuses_at_each_prompt(tmp_path, model, prompts):
    iterator = iter(prompts)
    statuses = []
    output = []

    class RecordingPresenter(CliChatPresenter):
        def prompt(self):
            statuses.append(self.prompt_status().context_used)
            return super().prompt()

    run_chat(
        model,
        tmp_path,
        "",
        "fake-model",
        presenter=RecordingPresenter(
            input_fn=lambda _prompt: next(iterator), output_fn=output.append
        ),
        tools_factory=create_workspace_tools,
        context_limit=1000,
    )
    return statuses, output


def test_a_successful_turn_without_measure_clears_the_stale_context(tmp_path):
    model = FakeChatModel(
        [
            AIMessage(content="un", usage_metadata=USAGE),
            AIMessage(content="deux, sans mesure"),
        ]
    )

    statuses, output = _statuses_at_each_prompt(
        tmp_path, model, ["a", "b", "/context", "/exit"]
    )

    assert statuses == [None, 500, None, None]
    assert any("aucune mesure" in line for line in output)


def test_a_failed_turn_keeps_the_last_measure_because_the_history_is_rolled_back(
    tmp_path,
):
    class FailingSecondTurn(FakeChatModel):
        def invoke(self, messages):
            if self.calls:
                self.calls.append(messages)
                raise RuntimeError("panne du modèle")
            self.calls.append(messages)
            return AIMessage(content="un", usage_metadata=USAGE)

    statuses, output = _statuses_at_each_prompt(
        tmp_path, FailingSecondTurn(), ["a", "b", "/context", "/exit"]
    )

    assert statuses == [None, 500, 500, 500]
    assert any(line.startswith("erreur>") for line in output)


def test_a_cancelled_turn_reports_and_stores_no_measure(tmp_path):
    holder = {}
    discarded = {"input_tokens": 800, "output_tokens": 100, "total_tokens": 900}
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {}, "id": "1"}],
        usage_metadata=discarded,
    )

    class CancelledSecondTurn(FakeChatModel):
        def invoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(content="un", usage_metadata=USAGE)
            holder["session"].cancel()
            return tool_call

    statuses = []
    output = []
    prompts = iter(["a", "b", "/context", "/exit"])

    class RecordingPresenter(CliChatPresenter):
        def prompt(self):
            statuses.append(self.prompt_status().context_used)
            return super().prompt()

    session = ChatSession(
        CancelledSecondTurn(),
        tmp_path,
        "",
        "fake-model",
        presenter=RecordingPresenter(
            input_fn=lambda _prompt: next(prompts), output_fn=output.append
        ),
        tools_factory=create_workspace_tools,
        context_limit=1000,
    )
    holder["session"] = session
    session.start()
    while session.handle(session._presenter.prompt()):
        pass

    assert "Génération annulée" in output
    # La mesure du tour annulé (900) n'est ni affichée ni gardée : l'historique est revenu
    # à l'état du premier tour (500).
    assert not any("900" in line for line in output)
    assert statuses == [None, 500, 500, 500]
    assert sum(line.startswith("contexte> 500") for line in output) == 2
