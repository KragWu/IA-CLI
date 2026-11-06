"""Interactive chat use case backed by the LangGraph control flow."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from time import perf_counter

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from iacli.services.agent_graph import AgentState, build_control_graph
from iacli.services.ports import ChatPresenter, ToolsFactory

MAX_TOOL_ITERATIONS = 25
# Part de la fenêtre de contexte à partir de laquelle l'utilisateur est alerté.
CONTEXT_WARNING_RATIO = 0.8


def evaluate_response(state: AgentState) -> bool:
    response = state["messages"][-1]
    if not isinstance(response, AIMessage):
        raise TypeError("Le modèle n'a pas terminé par un message assistant.")
    if response.tool_calls:
        raise ValueError("Le modèle a terminé avec des appels d'outils en attente.")
    if not response.content:
        raise ValueError("Le modèle a renvoyé une réponse vide.")
    return True


def run_chat(
    model: object,
    workspace: Path,
    instructions: str,
    model_name: str,
    *,
    presenter: ChatPresenter,
    tools_factory: ToolsFactory,
    context_limit: int | None = None,
) -> None:
    chat_presenter = presenter
    tools = tools_factory(
        workspace,
        chat_presenter.confirm_file_write,
        chat_presenter.confirm_shell_command,
        chat_presenter.show_action_timing,
    )
    graph = build_control_graph(
        model,
        tools,
        evaluate_response,
        max_iterations=MAX_TOOL_ITERATIONS,
        on_agent_timing=lambda elapsed: chat_presenter.show_action_timing(
            "modèle", elapsed
        ),
    )
    chat_presenter.start(workspace, model_name)
    messages = [
        SystemMessage(
            content=(
                "Tu es un assistant de programmation local. Réponds en français par défaut. "
                "Le répertoire de travail est le projet courant. Utilise les outils pour "
                "lire et rechercher dans le projet; toute modification de fichier et toute "
                "commande shell nécessitent une confirmation explicite de l'utilisateur. "
                "Quand tu dois utiliser un outil, appelle-le via l'interface d'outils; "
                "ne renvoie pas un appel d'outil sous forme de JSON dans ta réponse.\n\n"
                f"Instructions globales :\n{instructions}"
            )
        )
    ]

    while True:
        try:
            user_input = chat_presenter.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            return
        if user_input.casefold() in {"/exit", "/quit"}:
            return
        if not user_input:
            continue
        chat_presenter.start_progress()
        request_started = perf_counter()
        outcome = _run_turn(
            graph, messages, user_input, chat_presenter, request_started
        )
        messages = outcome.messages
        if outcome.error is not None:
            chat_presenter.show_error(str(outcome.error) or type(outcome.error).__name__)
        else:
            chat_presenter.show_response(outcome.response)
        if context_limit is not None and outcome.context_tokens is not None:
            chat_presenter.show_context_usage(outcome.context_tokens, context_limit)
            if outcome.context_tokens >= context_limit * CONTEXT_WARNING_RATIO:
                chat_presenter.show_context_warning(outcome.context_tokens, context_limit)


@dataclass
class _TurnOutcome:
    """Résultat d'un tour : l'historique à conserver et, selon le cas, la réponse ou l'erreur."""

    messages: list[BaseMessage]
    response: str = ""
    error: Exception | None = None
    context_tokens: int | None = None


def _run_turn(
    graph: Any,
    messages: list[BaseMessage],
    user_input: str,
    chat_presenter: ChatPresenter,
    request_started: float,
) -> _TurnOutcome:
    turn_input = [*messages, HumanMessage(content=user_input)]
    last_messages = turn_input
    produced_state = False
    seen_messages = len(messages) + 1
    try:
        try:
            for state in graph.stream(
                {"messages": turn_input},
                config={"max_concurrency": 1},
                stream_mode="values",
            ):
                produced_state = True
                last_messages = state["messages"]
                for message in last_messages[seen_messages:]:
                    _present_message(message, chat_presenter)
                seen_messages = max(seen_messages, len(last_messages))
        finally:
            chat_presenter.stop_progress()
            chat_presenter.show_total_duration(perf_counter() - request_started)

        if not produced_state:
            raise RuntimeError("Le graphe agent n'a produit aucun état.")
        response = last_messages[-1].content
        if not isinstance(response, str):
            raise TypeError("Le contenu de la réponse du modèle n'est pas du texte.")
    except KeyboardInterrupt:
        return _TurnOutcome(
            _without_incomplete_tail(last_messages),
            error=RuntimeError("Tour interrompu par l'utilisateur."),
            context_tokens=_context_tokens(last_messages, len(messages) + 1),
        )
    except Exception as error:  # noqa: BLE001 - un tour en échec ne doit pas fermer la session
        return _TurnOutcome(
            _without_incomplete_tail(last_messages),
            error=error,
            context_tokens=_context_tokens(last_messages, len(messages) + 1),
        )
    return _TurnOutcome(
        last_messages,
        response,
        context_tokens=_context_tokens(last_messages, len(messages) + 1),
    )


def _context_tokens(messages: list[BaseMessage], start: int) -> int | None:
    """Tokens occupés après le dernier appel au modèle du tour (messages[start:]),
    ou None si Ollama n'a rien renvoyé : la sortie sera rejouée au tour suivant."""
    for message in reversed(messages[start:]):
        if isinstance(message, AIMessage) and message.usage_metadata:
            return message.usage_metadata["total_tokens"]
    return None


def _without_incomplete_tail(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Retire la fin d'un tour en échec qui laisserait un historique invalide :
    un appel d'outil sans résultat ou une réponse vide du modèle."""
    kept = list(messages)
    while (
        kept
        and isinstance(kept[-1], AIMessage)
        and (kept[-1].tool_calls or not kept[-1].content)
    ):
        kept.pop()
    return kept


def _present_message(message: BaseMessage, chat_presenter: ChatPresenter) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        tool_names = ", ".join(tool_call["name"] for tool_call in message.tool_calls)
        chat_presenter.update_progress(f"Appel des outils : {tool_names}")
        for tool_call in message.tool_calls:
            chat_presenter.show_tool_call(tool_call["name"], tool_call["args"])
    elif isinstance(message, ToolMessage):
        chat_presenter.update_progress(
            f"Résultat de {message.name} reçu; l'agent poursuit"
        )
        chat_presenter.show_tool_result(message.name)
    elif isinstance(message, AIMessage):
        chat_presenter.update_progress("Préparation de la réponse")
