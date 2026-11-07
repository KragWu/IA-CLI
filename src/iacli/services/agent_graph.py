"""Orchestration LangGraph d'un agent qui appelle des outils (modèle et outils injectés)."""

import json
import re
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict, total=False):
    """État partagé par les nœuds du graphe.

    `messages` : historique (les nouveaux messages sont ajoutés, pas remplacés) ;
    `tool_iterations` : exécutions d'outils déjà faites pendant ce tour ;
    `evaluation` : résultat de l'évaluateur ; `error` : motif d'arrêt anticipé."""

    messages: Annotated[list[AnyMessage], add_messages]
    tool_iterations: int
    evaluation: object
    error: str


# Bloc de code markdown ```json ... ``` glissé par un petit modèle dans sa prose.
_JSON_FENCE = re.compile(r"```json[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)

Evaluator = Callable[[AgentState], object]
AgentTimingCallback = Callable[[float], None]


def format_tool_error(error: Exception) -> str:
    """Texte d'erreur renvoyé au modèle pour qu'il puisse corriger son appel.

    L'entrée standard fermée n'est pas une erreur de l'appel : elle est relancée
    pour arrêter le tour, sinon le modèle réessaierait en boucle."""
    if isinstance(error, EOFError):
        raise error
    return f"Erreur de l'outil : {error}"


class MaxToolIterationsError(RuntimeError):
    """Levée quand le modèle demande encore un outil alors que la limite d'exécutions est atteinte."""


def build_control_graph(
    model: Any,
    tools: Sequence[BaseTool],
    evaluator: Evaluator,
    *,
    max_iterations: int = 25,
    on_agent_timing: AgentTimingCallback | None = None,
):
    """Construit un graphe qui alterne entre le modèle (nœud `agent`) et ses outils (nœud `action`).

    `model` doit fournir `bind_tools` et `invoke`. Quand le modèle répond sans appel
    d'outil, le nœud `evaluator` appelle `evaluator(state)` : une exception de sa part
    fait échouer le tour. Au-delà de `max_iterations` exécutions d'outils, la
    demande suivante d'outil arrête le tour avec `MaxToolIterationsError`.
    `on_agent_timing` reçoit la durée (en secondes) de chaque appel au modèle.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations doit être supérieur ou égal à 1.")

    agent_model = model.bind_tools(tools)
    action_node = ToolNode(tools, handle_tool_errors=format_tool_error)
    tool_names = {tool.name for tool in tools}

    def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
        """Renvoie des tool calls si `text` ne contient que des objets JSON
        `{"name": ..., "arguments": {...}}` désignant des outils connus, sinon None."""
        decoder = json.JSONDecoder()
        position = 0
        payloads = []
        while position < len(text):
            while position < len(text) and text[position].isspace():
                position += 1
            if position == len(text):
                break
            try:
                payload, position = decoder.raw_decode(text, position)
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict):
                return None
            payloads.append(payload)

        if not payloads:
            return None

        calls = []
        for payload in payloads:
            name = payload.get("name")
            arguments = payload.get("arguments")
            if (
                not isinstance(name, str)
                or name not in tool_names
                or not isinstance(arguments, dict)
            ):
                return None
            calls.append(
                {
                    "name": name,
                    "args": arguments,
                    "id": str(uuid4()),
                    "type": "tool_call",
                }
            )
        return calls

    def normalize_text_tool_call(response: Any) -> Any:
        """Convertit un appel d'outil écrit en JSON dans le texte (brut ou dans un bloc
        ```json) en vrai tool call ; sinon renvoie la réponse inchangée.

        Certains petits modèles ne passent pas par le mécanisme de tool calling d'Ollama."""
        if not isinstance(response, AIMessage) or response.tool_calls:
            return response
        if not isinstance(response.content, str):
            return response
        content = response.content
        calls = parse_tool_calls(content)
        if calls is None:
            blocks = _JSON_FENCE.findall(content)
            if not blocks:
                return response
            calls = []
            for block in blocks:
                block_calls = parse_tool_calls(block)
                if block_calls is None:
                    return response
                calls.extend(block_calls)

        return AIMessage(
            content="",
            tool_calls=calls,
            id=response.id,
            response_metadata=response.response_metadata,
            additional_kwargs=response.additional_kwargs,
            usage_metadata=response.usage_metadata,
        )

    def agent(state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        try:
            response = normalize_text_tool_call(agent_model.invoke(state["messages"]))
        finally:
            if on_agent_timing is not None:
                on_agent_timing(perf_counter() - started)
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
        """Aiguillage : vers les outils si le modèle en demande (et que la limite n'est pas dépassée), sinon vers l'évaluateur."""
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
