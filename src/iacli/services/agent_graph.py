"""LangGraph orchestration for an injectable tool-calling agent."""

import json
from collections.abc import Callable, Sequence
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_iterations: int
    evaluation: object
    error: str


Evaluator = Callable[[AgentState], object]


class MaxToolIterationsError(RuntimeError):
    """Raised when the agent requests another tool after reaching its limit."""


def build_control_graph(
    model: Any,
    tools: Sequence[BaseTool],
    evaluator: Evaluator,
    *,
    max_iterations: int = 25,
):
    """Build a graph that alternates between an agent and its tools."""
    if max_iterations < 1:
        raise ValueError("max_iterations doit être supérieur ou égal à 1.")

    agent_model = model.bind_tools(tools)
    action_node = ToolNode(tools)
    tool_names = {tool.name for tool in tools}

    def normalize_text_tool_call(response: Any) -> Any:
        if not isinstance(response, AIMessage) or response.tool_calls:
            return response
        if not isinstance(response.content, str):
            return response
        decoder = json.JSONDecoder()
        content = response.content
        position = 0
        payloads = []
        while position < len(content):
            while position < len(content) and content[position].isspace():
                position += 1
            if position == len(content):
                break
            try:
                payload, position = decoder.raw_decode(content, position)
            except json.JSONDecodeError:
                return response
            if not isinstance(payload, dict):
                return response
            payloads.append(payload)

        if not payloads:
            return response

        calls = []
        for payload in payloads:
            name = payload.get("name")
            arguments = payload.get("arguments")
            if (
                not isinstance(name, str)
                or name not in tool_names
                or not isinstance(arguments, dict)
            ):
                return response
            calls.append(
                {
                    "name": name,
                    "args": arguments,
                    "id": str(uuid4()),
                    "type": "tool_call",
                }
            )

        return AIMessage(
            content="",
            tool_calls=calls,
            id=response.id,
            response_metadata=response.response_metadata,
        )

    def agent(state: AgentState) -> dict[str, Any]:
        response = normalize_text_tool_call(agent_model.invoke(state["messages"]))
        updates: dict[str, Any] = {
            "messages": [response],
            "tool_iterations": state.get("tool_iterations", 0),
        }
        if response.tool_calls and state.get("tool_iterations", 0) >= max_iterations:
            updates["error"] = (
                f"Limite de {max_iterations} exécutions d'outils atteinte."
            )
        return updates

    def route_after_agent(state: AgentState) -> str:
        if state.get("error"):
            return "evaluator"
        if state["messages"][-1].tool_calls:
            return "action"
        return "evaluator"

    def action(state: AgentState) -> dict[str, Any]:
        result = action_node.invoke(state)
        return {
            **result,
            "tool_iterations": state.get("tool_iterations", 0) + 1,
        }

    def evaluate(state: AgentState) -> dict[str, Any]:
        if error := state.get("error"):
            raise MaxToolIterationsError(error)
        return {"evaluation": evaluator(state)}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("action", action)
    graph.add_node("evaluator", evaluate)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"action": "action", "evaluator": "evaluator"},
    )
    graph.add_edge("action", "agent")
    graph.add_edge("evaluator", END)
    return graph.compile()
