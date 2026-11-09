"""Orchestration LangGraph d'un agent qui appelle des outils (modèle et outils injectés)."""

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from iacli.services.tool_call_parser import (
    MalformedToolCall,
    make_tool_call_feedback,
    normalize_text_tool_call,
)


class AgentState(TypedDict, total=False):
    """État partagé par les nœuds du graphe.

    `messages` : historique (les nouveaux messages sont ajoutés, pas remplacés) ;
    `tool_iterations` : exécutions d'outils déjà faites pendant ce tour ;
    `evaluation` : résultat de l'évaluateur ; `error` : exception qui arrête le tour (levée par l'évaluateur) ;
    `malformed_streak` : appels d'outil illisibles consécutifs (0 si le dernier
    appel du modèle était lisible) ; le modèle doit alors corriger son appel."""

    messages: Annotated[list[AnyMessage], add_messages]
    tool_iterations: int
    evaluation: object
    error: Exception
    malformed_streak: int


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


class MalformedToolCallsError(RuntimeError):
    """Levée quand le modèle n'arrive pas à corriger ses appels d'outil illisibles."""


def build_control_graph(
    model: Any,
    tools: Sequence[BaseTool],
    evaluator: Evaluator,
    *,
    max_iterations: int = 25,
    max_malformed_retries: int = 3,
    on_agent_timing: AgentTimingCallback | None = None,
):
    """Construit un graphe qui alterne entre le modèle (nœud `agent`) et ses outils (nœud `action`).

    `model` doit fournir `bind_tools` et `invoke`. Quand le modèle répond sans appel
    d'outil, le nœud `evaluator` appelle `evaluator(state)` : une exception de sa part
    fait échouer le tour. Au-delà de `max_iterations` exécutions d'outils, la
    demande suivante d'outil arrête le tour avec `MaxToolIterationsError`. Un appel
    d'outil illisible est renvoyé au modèle pour correction, sans compter parmi les
    exécutions ; après `max_malformed_retries` réessais consécutifs, le tour s'arrête
    avec `MalformedToolCallsError`.
    `on_agent_timing` reçoit la durée (en secondes) de chaque appel au modèle.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations doit être supérieur ou égal à 1.")
    if max_malformed_retries < 1:
        raise ValueError("max_malformed_retries doit être supérieur ou égal à 1.")

    agent_model = model.bind_tools(tools)
    action_node = ToolNode(tools, handle_tool_errors=format_tool_error)
    tool_names = {tool.name for tool in tools}

    def agent(state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        try:
            raw_response: BaseMessage = agent_model.invoke(state["messages"])
        finally:
            if on_agent_timing is not None:
                on_agent_timing(perf_counter() - started)
        response = normalize_text_tool_call(raw_response, tool_names)
        iterations = state.get("tool_iterations", 0)
        limit_error = MaxToolIterationsError(
            f"Limite de {max_iterations} exécutions d'outils atteinte."
        )

        if isinstance(response, MalformedToolCall):
            # Le modèle voit son appel raté et l'erreur, puis réessaie (borné par
            # `max_malformed_retries` pour ne pas boucler indéfiniment).
            streak = state.get("malformed_streak", 0) + 1
            updates: dict[str, Any] = {
                "messages": [raw_response, make_tool_call_feedback(response.error)],
                "tool_iterations": iterations,
                "malformed_streak": streak,
            }
            if streak > max_malformed_retries:
                updates["error"] = MalformedToolCallsError(
                    f"Le modèle a produit {streak} appels d'outil illisibles de suite."
                )
            return updates

        updates = {
            "messages": [response],
            "tool_iterations": iterations,
            "malformed_streak": 0,
        }
        if (
            isinstance(response, AIMessage)
            and response.tool_calls
            and iterations >= max_iterations
        ):
            updates["error"] = limit_error
        return updates

    def route_after_agent(state: AgentState) -> str:
        """Aiguillage : vers les outils si le modèle en demande (et que la limite n'est pas dépassée), sinon vers l'évaluateur."""
        if state.get("error"):
            return "evaluator"
        if state.get("malformed_streak"):
            return "agent"
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
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
            raise error
        return {"evaluation": evaluator(state)}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("action", action)
    graph.add_node("evaluator", evaluate)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"action": "action", "agent": "agent", "evaluator": "evaluator"},
    )
    graph.add_edge("action", "agent")
    graph.add_edge("evaluator", END)
    return graph.compile()
