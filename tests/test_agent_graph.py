import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from iacli.services.agent_graph import MaxToolIterationsError, build_control_graph


@tool
def read_file(path: str) -> str:
    """Read the contents of a file."""
    return f"contents of {path}"


class FakeChatModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = iter(responses)
        self.calls: list[list] = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return next(self.responses)


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
