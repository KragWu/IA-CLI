"""Interactive chat use case backed by the LangGraph control flow."""

from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.infrastructure.workspace_tools import create_workspace_tools
from iacli.services.agent_graph import AgentState, build_control_graph

MAX_TOOL_ITERATIONS = 25


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
    presenter: CliChatPresenter | None = None,
) -> None:
    chat_presenter = presenter or CliChatPresenter()
    tools = create_workspace_tools(
        workspace,
        chat_presenter.confirm_file_write,
        chat_presenter.confirm_shell_command,
    )
    graph = build_control_graph(
        model,
        tools,
        evaluate_response,
        max_iterations=MAX_TOOL_ITERATIONS,
    )
    chat_presenter.start(workspace, model_name)
    messages = [
        SystemMessage(
            content=(
                "Tu es un assistant de programmation local. Réponds en français par défaut. "
                "Le répertoire de travail est le projet courant. Utilise les outils pour "
                "lire et rechercher dans le projet; toute modification de fichier et toute "
                "commande shell nécessitent une confirmation explicite de l'utilisateur.\n\n"
                f"Instructions globales :\n{instructions}"
            )
        )
    ]

    while True:
        try:
            user_input = chat_presenter.prompt().strip()
        except EOFError:
            return
        if user_input.casefold() in {"/exit", "/quit"}:
            return
        if not user_input:
            continue
        result = graph.invoke(
            {"messages": [*messages, HumanMessage(content=user_input)]},
            config={"max_concurrency": 1},
        )
        messages = result["messages"]
        response = messages[-1].content
        if not isinstance(response, str):
            raise TypeError("Le contenu de la réponse du modèle n'est pas du texte.")
        chat_presenter.show_response(response)
