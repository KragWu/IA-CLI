import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from iacli.services.agent_graph import (
    MalformedToolCallsError,
    MaxToolIterationsError,
    build_control_graph,
)
from iacli.services.tool_call_parser import is_tool_call_feedback
from tests.fakes import FakeChatModel


@tool
def read_file(path: str) -> str:
    """Read the contents of a file."""
    return f"contents of {path}"


def test_graph_node_transitions():
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "README.md"},
                        "id": "read-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Le fichier contient le texte attendu."),
        ]
    )
    evaluated_states = []
    graph = build_control_graph(
        model,
        [read_file],
        lambda state: evaluated_states.append(state) or "verified",
    )

    result = graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(model.calls) == 2
    assert model.calls[1][-1] == tool_messages[0]
    assert tool_messages[0].content == "contents of README.md"
    assert tool_messages[0].tool_call_id == "read-1"
    assert result["tool_iterations"] == 1
    assert result["evaluation"] == "verified"
    assert len(evaluated_states) == 1


def test_agent_timing_callback_reports_each_model_call():
    model = FakeChatModel(
        [
            AIMessage(content='{"name":"read_file","arguments":{"path":"README.md"}}'),
            AIMessage(content="Réponse finale."),
        ]
    )
    timings = []
    graph = build_control_graph(
        model,
        [read_file],
        lambda _state: True,
        on_agent_timing=timings.append,
    )

    graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    assert len(timings) == 2
    assert all(duration >= 0 for duration in timings)


def test_graph_routes_json_text_tool_call_through_action_node():
    model = FakeChatModel(
        [
            AIMessage(content='{"name":"read_file","arguments":{"path":"README.md"}}'),
            AIMessage(content="Voici les dernières lignes du fichier."),
        ]
    )
    graph = build_control_graph(model, [read_file], lambda _state: True)

    result = graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(model.calls) == 2
    assert isinstance(model.calls[1][-1], ToolMessage)
    assert model.calls[1][-1].content == "contents of README.md"
    assert tool_messages[0].content == "contents of README.md"
    assert result["messages"][-1].content == "Voici les dernières lignes du fichier."


def test_graph_routes_multiple_json_text_tool_calls():
    model = FakeChatModel(
        [
            AIMessage(
                content=(
                    '{"name":"read_file","arguments":{"path":"README.md"}}\n'
                    '{"name":"read_file","arguments":{"path":"src/iacli/cli.py"}}'
                )
            ),
            AIMessage(content="Voici les premières lignes des deux fichiers."),
        ]
    )
    graph = build_control_graph(model, [read_file], lambda _state: True)

    result = graph.invoke({"messages": [HumanMessage(content="Lis deux fichiers")]})

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(model.calls) == 2
    assert len(tool_messages) == 2
    assert [message.content for message in tool_messages] == [
        "contents of README.md",
        "contents of src/iacli/cli.py",
    ]
    assert all(isinstance(message, ToolMessage) for message in model.calls[1][-2:])
    assert (
        result["messages"][-1].content
        == "Voici les premières lignes des deux fichiers."
    )


def test_graph_routes_json_tool_call_in_a_fenced_block_surrounded_by_prose():
    model = FakeChatModel(
        [
            AIMessage(
                content=(
                    "Voici le code :\n```java\nclass A {}\n```\n"
                    "Vous pouvez utiliser l'outil `read_file` :\n"
                    '```json\n{"name": "read_file", "arguments": {"path": "A.java"}}\n```\n'
                    "Dites-moi si vous voulez continuer !"
                )
            ),
            AIMessage(content="Fichier lu."),
        ]
    )
    graph = build_control_graph(model, [read_file], lambda _state: True)

    result = graph.invoke({"messages": [HumanMessage(content="Lis A.java")]})

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert [message.content for message in tool_messages] == ["contents of A.java"]
    assert result["messages"][-1].content == "Fichier lu."


def test_graph_ignores_fenced_json_that_is_not_a_known_tool_call():
    content = (
        "Exemple de configuration :\n"
        '```json\n{"name": "inconnu", "arguments": {"path": "A.java"}}\n```\n'
        '```json\n{"port": 8080}\n```'
    )
    model = FakeChatModel([AIMessage(content=content)])
    graph = build_control_graph(model, [read_file], lambda _state: True)

    result = graph.invoke({"messages": [HumanMessage(content="Montre un exemple")]})

    assert not any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["messages"][-1].content == content


def test_graph_completion():
    model = FakeChatModel([AIMessage(content="La réponse est complète.")])
    graph = build_control_graph(
        model,
        [read_file],
        lambda state: state["messages"][-1].content,
    )

    result = graph.invoke({"messages": [HumanMessage(content="Bonjour")]})

    assert len(model.calls) == 1
    assert result["tool_iterations"] == 0
    assert result["evaluation"] == "La réponse est complète."
    assert "error" not in result


def test_max_iterations_safety():
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "README.md"},
                        "id": f"read-{index}",
                        "type": "tool_call",
                    }
                ],
            )
            for index in range(4)
        ]
    )
    graph = build_control_graph(
        model,
        [read_file],
        lambda state: "evaluated",
        max_iterations=2,
    )

    with pytest.raises(
        MaxToolIterationsError,
        match=r"Limite de 2 exécutions d'outils atteinte\.",
    ):
        graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    assert len(model.calls) == 3


@tool
def failing_tool(path: str) -> str:
    """Always fail with a domain error."""
    raise FileNotFoundError(f"Fichier introuvable : {path}")


def test_tool_exception_is_returned_to_the_model_instead_of_aborting():
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "failing_tool",
                        "args": {"path": "absent.md"},
                        "id": "fail-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Le fichier n'existe pas."),
        ]
    )
    graph = build_control_graph(model, [failing_tool], lambda _state: "verified")

    result = graph.invoke({"messages": [HumanMessage(content="Lis absent.md")]})

    tool_message = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert tool_message.tool_call_id == "fail-1"
    assert tool_message.status == "error"
    assert "Fichier introuvable : absent.md" in tool_message.content
    assert model.calls[1][-1] == tool_message
    assert result["messages"][-1].content == "Le fichier n'existe pas."


@tool
def closed_input_tool(path: str) -> str:
    """Fail as if stdin was closed during a confirmation."""
    raise EOFError


def test_closed_input_during_a_tool_is_not_returned_to_the_model():
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "closed_input_tool",
                        "args": {"path": "a.txt"},
                        "id": "eof-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Je réessaie."),
        ]
    )
    graph = build_control_graph(model, [closed_input_tool], lambda _state: "verified")

    with pytest.raises(EOFError):
        graph.invoke({"messages": [HumanMessage(content="Écris a.txt")]})

    assert len(model.calls) == 1


def test_json_text_tool_call_conversion_keeps_usage_and_extra_fields():
    usage = {"input_tokens": 900, "output_tokens": 100, "total_tokens": 1000}
    model = FakeChatModel(
        [
            AIMessage(
                content='{"name":"read_file","arguments":{"path":"README.md"}}',
                usage_metadata=usage,
                additional_kwargs={"origine": "texte"},
            ),
            AIMessage(content="Terminé."),
        ]
    )
    graph = build_control_graph(model, [read_file], lambda _state: "verified")

    result = graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    converted = next(
        message
        for message in result["messages"]
        if isinstance(message, AIMessage) and message.tool_calls
    )
    assert converted.usage_metadata == usage
    assert converted.additional_kwargs == {"origine": "texte"}


def test_graph_routes_tool_call_tag_through_action_node():
    model = FakeChatModel(
        [
            AIMessage(
                content=(
                    '<tool_call>{"name": "read_file", '
                    '"arguments": {"path": "README.md"}}</tool_call>'
                )
            ),
            AIMessage(content="Fichier lu."),
        ]
    )
    graph = build_control_graph(model, [read_file], lambda _state: True)

    result = graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert [message.content for message in tool_messages] == ["contents of README.md"]
    assert result["messages"][-1].content == "Fichier lu."


def test_graph_reports_unknown_tool_in_tag_to_the_model():
    model = FakeChatModel(
        [
            AIMessage(
                content='<tool_call>{"name": "inconnu", "arguments": {}}</tool_call>'
            ),
            AIMessage(content="Je me corrige."),
        ]
    )
    graph = build_control_graph(model, [read_file], lambda _state: True)

    result = graph.invoke({"messages": [HumanMessage(content="Fais quelque chose")]})

    tool_message = model.calls[1][-1]
    assert isinstance(tool_message, ToolMessage)
    assert "inconnu" in tool_message.content
    assert tool_message.status == "error"
    assert result["messages"][-1].content == "Je me corrige."


def test_malformed_tool_call_handling_asks_the_model_to_fix_its_call():
    broken = AIMessage(content='<tool_call>{"name": "read_file", </tool_call>')
    model = FakeChatModel(
        [
            broken,
            AIMessage(
                content=(
                    '<tool_call>{"name": "read_file", '
                    '"arguments": {"path": "README.md"}}</tool_call>'
                )
            ),
            AIMessage(content="Fichier lu."),
        ]
    )
    graph = build_control_graph(model, [read_file], lambda _state: True)

    result = graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    feedback = model.calls[1][-1]
    assert isinstance(feedback, HumanMessage)
    assert "JSON" in feedback.content
    assert model.calls[1][-2].content == broken.content
    assert [
        message.content
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ] == ["contents of README.md"]
    assert result["messages"][-1].content == "Fichier lu."
    assert result["tool_iterations"] == 1


BROKEN = "<tool_call>{oups</tool_call>"
VALID_CALL = (
    '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>'
)


def test_malformed_tool_call_feedback_is_marked_as_not_coming_from_the_user():
    model = FakeChatModel([AIMessage(content=BROKEN), AIMessage(content="Terminé.")])
    graph = build_control_graph(model, [read_file], lambda _state: True)

    graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    feedback = model.calls[1][-1]
    assert isinstance(feedback, HumanMessage)
    assert is_tool_call_feedback(feedback)
    assert not is_tool_call_feedback(HumanMessage(content="Lis README.md"))


def test_malformed_tool_call_loop_is_stopped_by_its_own_retry_limit():
    model = FakeChatModel([AIMessage(content=BROKEN) for _ in range(10)])
    graph = build_control_graph(model, [read_file], lambda _state: True)

    with pytest.raises(MalformedToolCallsError, match="4 appels d'outil illisibles"):
        graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    assert len(model.calls) == 4  # 3 réessais autorisés par défaut


def test_malformed_tool_call_retry_limit_is_configurable():
    model = FakeChatModel([AIMessage(content=BROKEN) for _ in range(10)])
    graph = build_control_graph(
        model, [read_file], lambda _state: True, max_malformed_retries=1
    )

    with pytest.raises(MalformedToolCallsError):
        graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    assert len(model.calls) == 2


def test_malformed_tool_call_retries_do_not_use_the_tool_iteration_budget():
    model = FakeChatModel(
        [
            AIMessage(content=VALID_CALL),
            AIMessage(content=BROKEN),
            AIMessage(content="Terminé."),
        ]
    )
    graph = build_control_graph(
        model, [read_file], lambda _state: True, max_iterations=1
    )

    result = graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    assert result["tool_iterations"] == 1
    assert result["messages"][-1].content == "Terminé."


def test_a_valid_response_resets_the_malformed_tool_call_count():
    model = FakeChatModel(
        [
            AIMessage(content=BROKEN),
            AIMessage(content=VALID_CALL),
            AIMessage(content=BROKEN),
            AIMessage(content="Terminé."),
        ]
    )
    graph = build_control_graph(
        model, [read_file], lambda _state: True, max_malformed_retries=1
    )

    result = graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    assert result["messages"][-1].content == "Terminé."


@pytest.mark.parametrize("option", ["max_iterations", "max_malformed_retries"])
def test_graph_rejects_limits_below_one(option):
    with pytest.raises(ValueError, match=option):
        build_control_graph(
            FakeChatModel(), [read_file], lambda _state: True, **{option: 0}
        )


CUT_BY_TOKEN_LIMIT = AIMessage(
    content='<tool_call>{"name": "read_file", "arguments": {"path": "READ',
    response_metadata={"done_reason": "length"},
)


def test_truncated_tool_call_tells_the_model_to_make_a_shorter_call():
    model = FakeChatModel([CUT_BY_TOKEN_LIMIT, AIMessage(content="Terminé.")])
    graph = build_control_graph(model, [read_file], lambda _state: True)

    graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    feedback = model.calls[1][-1]
    assert is_tool_call_feedback(feedback)
    assert "limite de tokens" in feedback.content


def test_repeated_truncated_tool_calls_end_with_a_token_limit_error():
    model = FakeChatModel([CUT_BY_TOKEN_LIMIT for _ in range(10)])
    graph = build_control_graph(
        model, [read_file], lambda _state: True, max_malformed_retries=2
    )

    with pytest.raises(MalformedToolCallsError, match="limite de tokens") as error:
        graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    assert "num_ctx" in str(error.value)
    assert len(model.calls) == 3


def test_tool_call_cut_right_after_the_opening_tag_is_not_a_final_answer():
    model = FakeChatModel(
        [
            AIMessage(
                content="<tool_call>", response_metadata={"done_reason": "length"}
            ),
            AIMessage(content="Terminé."),
        ]
    )
    graph = build_control_graph(model, [read_file], lambda _state: True)

    result = graph.invoke({"messages": [HumanMessage(content="Lis README.md")]})

    assert is_tool_call_feedback(model.calls[1][-1])
    assert "limite de tokens" in model.calls[1][-1].content
    assert result["messages"][-1].content == "Terminé."


def test_final_answer_quoting_a_tool_call_in_code_is_not_executed():
    answer = (
        'Le format est `<tool_call>{"name": "read_file", '
        '"arguments": {"path": "README.md"}}</tool_call>`.'
    )
    model = FakeChatModel([AIMessage(content=answer)])
    graph = build_control_graph(model, [read_file], lambda _state: True)

    result = graph.invoke({"messages": [HumanMessage(content="Quel format ?")]})

    assert len(model.calls) == 1
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["messages"][-1].content == answer
