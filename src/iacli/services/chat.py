"""Cas d'usage du chat interactif, piloté par le graphe LangGraph de `agent_graph`."""

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from time import perf_counter
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from iacli.domain.commands import COMMANDS, InputKind, SlashCommand, parse_input
from iacli.services.agent_graph import AgentState, build_control_graph
from iacli.services.ports import ChatPresenter, ToolsFactory
from iacli.services.tool_call_parser import is_tool_call_feedback

# Nombre maximal d'exécutions d'outils par tour avant d'abandonner (évite les boucles).
MAX_TOOL_ITERATIONS = 25
# Part de la fenêtre de contexte à partir de laquelle l'utilisateur est alerté.
CONTEXT_WARNING_RATIO = 0.8


def evaluate_response(state: AgentState) -> bool:
    """Évaluateur de fin de tour : accepte la réponse ou lève une exception.

    Refuse un dernier message qui n'est pas de l'assistant, qui garde des appels
    d'outils en attente ou dont le contenu est vide."""
    response = state["messages"][-1]
    if not isinstance(response, AIMessage):
        raise TypeError("Le modèle n'a pas terminé par un message assistant.")
    if response.tool_calls:
        raise ValueError("Le modèle a terminé avec des appels d'outils en attente.")
    if not response.content:
        raise ValueError("Le modèle a renvoyé une réponse vide.")
    return True


class ChatSession:
    """Une conversation : l'historique, le graphe agent et la gestion des commandes.

    La session ne lit jamais la saisie : c'est à l'appelant (boucle de `run_chat`, ou une
    interface pilotée par événements) de lui transmettre chaque message avec `handle`.
    `cancel` peut être appelé depuis un autre thread pour interrompre le tour en cours.

    `model` doit supporter `bind_tools` (ex. `ChatOllama`). `workspace` est le projet
    courant, `instructions` le contenu d'`IACLI.md` ajouté à l'invite système et
    `context_limit` la fenêtre de contexte en tokens (None : pas d'affichage).
    """

    def __init__(
        self,
        model: object,
        workspace: Path,
        instructions: str,
        model_name: str,
        *,
        presenter: ChatPresenter,
        tools_factory: ToolsFactory,
        context_limit: int | None = None,
    ) -> None:
        self._presenter = presenter
        self._workspace = workspace
        self._model_name = model_name
        self._context_limit = context_limit
        tools = tools_factory(
            workspace,
            presenter.confirm_file_write,
            presenter.confirm_shell_command,
            presenter.show_action_timing,
        )
        self._graph = build_control_graph(
            model,
            tools,
            evaluate_response,
            max_iterations=MAX_TOOL_ITERATIONS,
            on_agent_timing=lambda elapsed: presenter.show_action_timing(
                "modèle", elapsed
            ),
        )
        # Historique conservé en mémoire pour toute la session (jamais écrit sur disque).
        self._initial_messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    "Tu es un assistant de programmation local. Réponds en français par défaut. "
                    "Le répertoire de travail est le projet courant. Tu disposes des outils "
                    "suivants :\n"
                    "- read_file : lire un fichier du projet ;\n"
                    "- list_files : lister les fichiers d'un dossier ;\n"
                    "- search_files : rechercher du texte dans le projet ;\n"
                    "- write_file : créer un fichier ou remplacer tout son contenu ;\n"
                    "- edit_file : remplacer un bloc de texte exact (old_string, qui doit "
                    "apparaître une seule fois) par new_string dans un fichier existant ;\n"
                    "- run_shell : exécuter une commande shell.\n"
                    "write_file, edit_file et run_shell demandent une confirmation "
                    "explicite de l'utilisateur, qui voit le diff ou la commande avant de "
                    "valider. Pour créer un fichier, appelle write_file ; pour modifier un "
                    "fichier existant, préfère edit_file (lis le fichier d'abord pour "
                    "copier old_string exactement) et garde write_file pour le réécrire "
                    "en entier. Quand l'utilisateur demande de créer ou de modifier un "
                    "fichier, utilise ces outils : ne colle pas de code dans le chat à la "
                    "place, et ne dis jamais que tu ne peux pas créer de fichiers ni "
                    "exécuter de commandes. "
                    "Quand tu dois utiliser un outil, appelle-le via l'interface d'outils; "
                    "ne renvoie pas un appel d'outil sous forme de JSON dans ta réponse.\n\n"
                    f"Instructions globales :\n{instructions}"
                )
            )
        ]
        self._messages = list(self._initial_messages)
        self._last_context_tokens: int | None = None
        self._cancel_requested = Event()
        # `_handling` : une saisie est en cours de traitement. Il évite de perdre une annulation
        # demandée juste après `handle` (autre thread) et d'en garder une venue hors traitement.
        self._state_lock = Lock()
        self._handling = False

    def start(self) -> None:
        """Affiche l'en-tête de la conversation."""
        self._presenter.start(self._workspace, self._model_name)

    def cancel(self) -> None:
        """Demande l'arrêt du tour en cours, à la prochaine étape du graphe.

        Utilisable depuis n'importe quel thread ; sans effet quand aucune saisie n'est en cours
        de traitement (`handle`). Un appel déjà lancé (modèle, commande shell) n'est pas
        interrompu de force."""
        with self._state_lock:
            if self._handling:
                self._cancel_requested.set()

    def handle(self, user_input: str) -> bool:
        """Traite une saisie (message ou slash-command) ; False quand la session doit se terminer."""
        with self._state_lock:
            self._handling = True
            self._cancel_requested.clear()
        try:
            return self._dispatch(user_input)
        finally:
            with self._state_lock:
                self._handling = False

    def _dispatch(self, user_input: str) -> bool:
        if not user_input.strip():
            return True
        parsed = parse_input(user_input)
        if parsed.kind is InputKind.MESSAGE:
            self._run_turn(_clean_message(user_input))
            return True
        command = parsed.command
        if command is None:
            self._presenter.show_error(
                f"Commande inconnue : {parsed.word}. "
                "Tapez /help pour la liste des commandes."
            )
            return True
        if parsed.kind is InputKind.COMMAND_WITH_ARGUMENTS:
            self._presenter.show_error(
                f"La commande {command.name} n'accepte pas d'argument : "
                "le reste de la saisie n'a pas été envoyé."
            )
            return True
        return self._handle_command(command)

    def _handle_command(self, command: SlashCommand) -> bool:
        presenter = self._presenter
        if command.name in {"/exit", "/quit"}:
            return False
        if command.name == "/help":
            presenter.show_help(COMMANDS)
        elif command.name == "/clear":
            self._messages = list(self._initial_messages)
            self._last_context_tokens = None
            presenter.clear_context_usage()
            presenter.show_notice("Historique effacé")
        elif command.name == "/context":
            if self._context_limit is None or self._last_context_tokens is None:
                presenter.show_notice("contexte> aucune mesure pour l'instant")
            else:
                presenter.show_context_usage(
                    self._last_context_tokens, self._context_limit
                )
        return True

    def _run_turn(self, user_input: str) -> None:
        presenter = self._presenter
        presenter.start_progress()
        outcome = _run_turn(
            self._graph,
            self._messages,
            user_input,
            presenter,
            perf_counter(),
            self._cancel_requested,
        )
        self._messages = outcome.messages
        if outcome.context_tokens is not None:
            self._last_context_tokens = outcome.context_tokens
        elif not outcome.cancelled and outcome.error is None:
            # Le tour a réussi mais Ollama n'a rien mesuré : l'historique a grandi, l'ancien
            # chiffre sous-estimerait le contexte. (Un tour échoué ou annulé est retiré de
            # l'historique, donc la dernière mesure reste exacte.)
            self._last_context_tokens = None
            presenter.clear_context_usage()
        if outcome.cancelled:
            presenter.show_cancelled()
        elif outcome.error is not None:
            presenter.show_error(str(outcome.error) or type(outcome.error).__name__)
        else:
            presenter.show_response(outcome.response)
        limit = self._context_limit
        if limit is not None and outcome.context_tokens is not None:
            presenter.show_context_usage(outcome.context_tokens, limit)
            if outcome.context_tokens >= limit * CONTEXT_WARNING_RATIO:
                presenter.show_context_warning(outcome.context_tokens, limit)


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
    """Boucle de conversation console : une saisie lue sur le presenter, un tour par message.

    La boucle s'arrête sur `/exit`, `/quit`, une fin de fichier ou Ctrl+C à l'invite ;
    une erreur pendant un tour est affichée sans fermer la session.
    """
    session = ChatSession(
        model,
        workspace,
        instructions,
        model_name,
        presenter=presenter,
        tools_factory=tools_factory,
        context_limit=context_limit,
    )
    session.start()
    while True:
        try:
            user_input = presenter.prompt()
        except (EOFError, KeyboardInterrupt):
            return
        if not session.handle(user_input):
            return


def _is_complete_answer(state: AgentState) -> bool:
    """True quand le modèle vient de produire sa réponse finale : plus d'appel d'outil à
    exécuter, aucun appel illisible à corriger. Annuler à ce stade jetterait un travail
    terminé, alors qu'il ne reste qu'à vérifier la réponse."""
    if state.get("error") or state.get("malformed_streak"):
        return False
    last_message = state["messages"][-1]
    return isinstance(last_message, AIMessage) and not last_message.tool_calls


def _clean_message(text: str) -> str:
    """Retire les lignes vides du début et les blancs de la fin d'un message, sans toucher à
    l'indentation de sa première ligne (un bloc de code ou du YAML collé doit rester aligné)."""
    lines = text.splitlines(keepends=True)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "".join(lines).rstrip()


class _TurnCancelled(Exception):
    """Annulation demandée par `ChatSession.cancel` (équivalent d'un Ctrl+C)."""


@dataclass
class _TurnOutcome:
    """Résultat d'un tour : l'historique à conserver et, selon le cas, la réponse ou l'erreur."""

    messages: list[BaseMessage]
    response: str = ""
    error: Exception | None = None
    cancelled: bool = False
    context_tokens: int | None = None


def _run_turn(
    graph: Any,
    messages: list[BaseMessage],
    user_input: str,
    chat_presenter: ChatPresenter,
    request_started: float,
    cancel_requested: Event,
) -> _TurnOutcome:
    """Exécute un tour (une demande de l'utilisateur) et en renvoie le résultat.

    Les messages produits sont présentés au fil de l'eau. Toute erreur ou fermeture de
    l'entrée est capturée dans `_TurnOutcome.error`, et Ctrl+C dans `_TurnOutcome.cancelled`,
    pour que la session continue.
    """
    turn_input = [*messages, HumanMessage(content=user_input)]
    last_messages = turn_input
    produced_state = False
    seen_messages = len(messages) + 1
    failure: Exception | None = None
    cancelled = False
    try:
        try:
            for state in graph.stream(
                {"messages": turn_input},
                config={"max_concurrency": 1},
                stream_mode="values",
            ):
                produced_state = True
                last_messages = state["messages"]
                _present_messages(
                    last_messages[seen_messages:],
                    chat_presenter,
                    will_retry=not state.get("error"),
                )
                seen_messages = max(seen_messages, len(last_messages))
                if cancel_requested.is_set() and not _is_complete_answer(state):
                    raise _TurnCancelled
        finally:
            chat_presenter.stop_progress()
            chat_presenter.show_total_duration(perf_counter() - request_started)

        if not produced_state:
            raise RuntimeError("Le graphe agent n'a produit aucun état.")
        content = last_messages[-1].content
        if not isinstance(content, str):
            raise TypeError("Le contenu de la réponse du modèle n'est pas du texte.")
        response = content
    except (KeyboardInterrupt, _TurnCancelled):
        cancelled = True
    except EOFError:
        failure = RuntimeError("L'entrée standard a été fermée pendant le tour.")
    except Exception as error:
        failure = error

    context_tokens = _context_tokens(last_messages, len(messages) + 1)
    if cancelled or failure is not None:
        return _TurnOutcome(
            _without_incomplete_tail(last_messages, turn_start=len(messages)),
            error=failure,
            cancelled=cancelled,
            context_tokens=context_tokens,
        )
    return _TurnOutcome(
        _without_tool_call_feedback(last_messages, start=len(messages) + 1),
        response,
        context_tokens=context_tokens,
    )


def _context_tokens(messages: list[BaseMessage], start: int) -> int | None:
    """Tokens occupés après le dernier appel au modèle du tour (messages[start:]),
    ou None si Ollama n'a rien renvoyé : la sortie sera rejouée au tour suivant."""
    for message in reversed(messages[start:]):
        if isinstance(message, AIMessage) and message.usage_metadata:
            return message.usage_metadata["total_tokens"]
    return None


def _without_tool_call_feedback(
    messages: list[BaseMessage], start: int
) -> list[BaseMessage]:
    """Retire, à partir de `start`, les appels d'outil illisibles du modèle et les
    corrections qui leur répondaient : le modèle les a vus pendant le tour, mais les
    rejouer aux tours suivants gaspille du contexte et l'incite à copier le format raté."""
    kept = list(messages[:start])
    for message in messages[start:]:
        if (
            is_tool_call_feedback(message)
            and len(kept) > start
            and isinstance(kept[-1], AIMessage)
        ):
            kept.pop()
        else:
            kept.append(message)
    return kept


def _without_incomplete_tail(
    messages: list[BaseMessage], turn_start: int
) -> list[BaseMessage]:
    """Retire la fin d'un tour en échec qui laisserait un historique invalide :
    un appel d'outil sans résultat ou une réponse vide du modèle, après avoir retiré
    les échanges d'appels d'outil illisibles. Si le tour n'a rien produit de plus que
    la demande (index turn_start), celle-ci est retirée aussi : elle resterait sans
    réponse et précéderait la demande suivante."""
    kept = _without_tool_call_feedback(messages, start=turn_start + 1)
    while (
        kept
        and isinstance(kept[-1], AIMessage)
        and (kept[-1].tool_calls or not kept[-1].content)
    ):
        kept.pop()
    if len(kept) == turn_start + 1:
        kept.pop()
    return kept


def _present_messages(
    messages: list[BaseMessage],
    chat_presenter: ChatPresenter,
    *,
    will_retry: bool = True,
) -> None:
    """Présente les nouveaux messages du graphe, dans l'ordre.

    Un appel d'outil illisible n'est pas annoncé comme une réponse : seule la
    correction qui le suit informe l'utilisateur que l'agent réessaie. Quand le tour
    est déjà en erreur (`will_retry` faux), il n'y aura pas de réessai : on ne l'annonce pas."""
    for index, message in enumerate(messages):
        has_next = index + 1 < len(messages)
        if has_next and is_tool_call_feedback(messages[index + 1]):
            continue  # appel raté : annoncé par sa correction, pas comme une réponse
        if not will_retry and is_tool_call_feedback(message):
            continue
        _present_message(message, chat_presenter)


def _present_message(message: BaseMessage, chat_presenter: ChatPresenter) -> None:
    """Traduit un message du graphe en mise à jour de progression et en lignes d'outils."""
    if isinstance(message, AIMessage) and message.tool_calls:
        tool_names = ", ".join(tool_call["name"] for tool_call in message.tool_calls)
        chat_presenter.update_progress(f"Appel des outils : {tool_names}")
        for tool_call in message.tool_calls:
            chat_presenter.show_tool_call(tool_call["name"], tool_call["args"])
    elif isinstance(message, ToolMessage):
        chat_presenter.update_progress(
            f"Résultat de {message.name} reçu; l'agent poursuit"
        )
        chat_presenter.show_tool_result(message.name or "")
    elif is_tool_call_feedback(message):
        chat_presenter.update_progress("Appel d'outil illisible ; l'agent réessaie")
    elif isinstance(message, AIMessage):
        chat_presenter.update_progress("Préparation de la réponse")
