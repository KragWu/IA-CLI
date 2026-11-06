import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from iacli.services.agent_graph import MaxToolIterationsError, build_control_graph
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

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
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
            AIMessage(
                content='{"name":"read_file","arguments":{"path":"README.md"}}'
            ),
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
    assert result["messages"][-1].content == "Voici les premières lignes des deux fichiers."


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
        match="Limite de 2 exécutions d'outils atteinte.",
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
