"""Reconnaissance des appels d'outils écrits dans le texte d'un modèle (logique pure).

Certains petits modèles locaux n'utilisent pas le tool calling natif d'Ollama et
écrivent leur appel en JSON dans leur réponse."""

import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# Bloc de code markdown ```json ... ``` glissé par un petit modèle dans sa prose.
_JSON_FENCE = re.compile(r"```json[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Format Hermes/Qwen : <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"

# Prochain élément qui compte pour le balayage : une balise ouvrante ou un délimiteur
# de code markdown (bloc ``` ou extrait `...`).
_NEXT_TOKEN = re.compile(re.escape(_TOOL_CALL_OPEN) + r"|```|`")


def _make_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": arguments, "id": str(uuid4()), "type": "tool_call"}


_TAG_EXPECTED_FORMAT = (
    'Renvoie un JSON valide de la forme {"name": "<outil>", "arguments": {...}} '
    "entre <tool_call> et </tool_call>."
)


@dataclass(frozen=True)
class MalformedToolCall:
    """Appel d'outil manifestement raté (balise `<tool_call>` illisible).

    `error` est un message destiné au modèle pour qu'il corrige son appel ;
    `truncated` indique que la réponse a été coupée par la limite de tokens."""

    error: str
    truncated: bool = False


_FEEDBACK_KEY = "iacli_tool_call_feedback"


def make_tool_call_feedback(error: str) -> HumanMessage:
    """Message qui renvoie au modèle l'erreur de son appel illisible.

    Il reste un `HumanMessage` (certains modèles locaux n'acceptent un message système
    qu'en début de conversation) mais est marqué : il ne vient pas de l'utilisateur."""
    return HumanMessage(content=error, additional_kwargs={_FEEDBACK_KEY: True})


def is_tool_call_feedback(message: BaseMessage) -> bool:
    """Indique si `message` a été créé par `make_tool_call_feedback`."""
    return isinstance(message, HumanMessage) and bool(
        message.additional_kwargs.get(_FEEDBACK_KEY)
    )


_TRUNCATED_REASON = (
    "ta réponse a été coupée par la limite de tokens avant la fin de l'appel. "
    "Fais un appel plus court (par exemple, découpe un gros write_file en "
    "plusieurs écritures)."
)


def truncated_series_message(streak: int) -> str:
    return (
        f"Les réponses du modèle ont été coupées par la limite de tokens {streak} fois "
        "de suite pendant un appel d'outil. Augmentez `num_ctx` dans config.toml ou "
        "demandez une action plus petite."
    )


def _error_message(reason: str) -> str:
    return (
        f"Ton appel d'outil <tool_call> est invalide : {reason} {_TAG_EXPECTED_FORMAT}"
    )


def _truncated() -> MalformedToolCall:
    return MalformedToolCall(_error_message(_TRUNCATED_REASON), truncated=True)


def _malformed(reason: str, *, truncated: bool = False) -> MalformedToolCall:
    """Un appel coupé par la limite de tokens se corrige autrement qu'une erreur de syntaxe."""
    if truncated:
        return _truncated()
    return MalformedToolCall(_error_message(reason))


def _skip_spaces(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _end_of_markdown_code(text: str, delimiter: re.Match[str]) -> int:
    """Position après le code markdown qui commence par `delimiter`, ou juste après le
    délimiteur s'il n'est jamais refermé (un bloc ``` peut couvrir plusieurs lignes, un
    extrait `...` reste sur sa ligne) : ce n'est alors que du texte ordinaire."""
    marker = delimiter.group()
    if marker == "```":
        limit = len(text)
    else:
        newline = text.find("\n", delimiter.end())
        limit = len(text) if newline == -1 else newline
    closing = text.find(marker, delimiter.end(), limit)
    return delimiter.end() if closing == -1 else closing + len(marker)


def _parse_tagged_tool_calls(
    text: str, *, truncated: bool = False
) -> list[dict[str, Any]] | MalformedToolCall | None:
    """Renvoie les tool calls des balises `<tool_call>` de `text`, un `MalformedToolCall`
    si l'une est invalide, None s'il n'y en a aucune. Le nom n'est pas confronté aux
    outils connus : `ToolNode` renverra lui-même une erreur au modèle pour un outil
    inconnu.

    Une balise n'est une tentative d'appel que si du JSON (`{` ou `[`) la suit ; sinon
    c'est une simple mention dans la prose, ignorée. Une balise citée dans du code
    markdown (bloc ``` ou extrait `...`) est un exemple, pas un appel : elle est ignorée
    aussi. Le texte est balayé dans l'ordre et le JSON d'une vraie balise est lu par un
    décodeur (non par une regex ou un masquage) : ses chaînes peuvent donc contenir
    `</tool_call>` ou du markdown.

    `truncated` : la génération s'est arrêtée sur la limite de tokens ; un JSON coupé
    ou une balise non refermée sont alors signalés comme tronqués."""
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    position = 0
    while (token := _NEXT_TOKEN.search(text, position)) is not None:
        if token.group() != _TOOL_CALL_OPEN:
            position = _end_of_markdown_code(text, token)
            continue
        start = _skip_spaces(text, token.end())
        if truncated and start == len(text):
            # Coupé juste après la balise ouvrante : rien ne dit encore que c'est du JSON.
            return _truncated()
        if text[start : start + 1] not in ("{", "["):
            position = start
            continue
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError as error:
            return _malformed(f"JSON illisible ({error}).", truncated=truncated)
        end = _skip_spaces(text, end)
        if not text.startswith(_TOOL_CALL_CLOSE, end):
            return _malformed(
                "une balise <tool_call> n'est pas refermée par </tool_call>.",
                truncated=truncated,
            )
        position = end + len(_TOOL_CALL_CLOSE)
        if not isinstance(payload, dict):
            return _malformed('le JSON doit être un objet avec "name" et "arguments".')
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str):
            return _malformed('le champ "name" (texte) est obligatoire.')
        if not isinstance(arguments, dict):
            return _malformed('le champ "arguments" doit être un objet JSON.')
        calls.append(_make_tool_call(name, arguments))
    return calls or None


def parse_tool_calls(
    text: str, tool_names: Collection[str]
) -> list[dict[str, Any]] | None:
    """Renvoie des tool calls si `text` ne contient que des objets JSON
    `{"name": ..., "arguments": {...}}` désignant des outils connus, sinon None."""
    decoder = json.JSONDecoder()
    position = 0
    payloads = []
    while position < len(text):
        position = _skip_spaces(text, position)
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
        calls.append(_make_tool_call(name, arguments))
    return calls


def normalize_text_tool_call(
    response: BaseMessage, tool_names: Collection[str]
) -> BaseMessage | MalformedToolCall:
    """Convertit un appel d'outil écrit dans le texte (JSON brut, bloc ```json ou balises
    <tool_call>) en vrai tool call ; sinon renvoie la réponse inchangée.

    Une balise `<tool_call>` illisible donne un `MalformedToolCall` : le texte brut ne
    doit pas devenir la réponse finale. Un JSON invalide hors balise reste du texte."""
    if not isinstance(response, AIMessage) or response.tool_calls:
        return response
    if not isinstance(response.content, str):
        return response
    content = response.content
    truncated = response.response_metadata.get("done_reason") == "length"
    calls = _parse_tagged_tool_calls(content, truncated=truncated)
    if isinstance(calls, MalformedToolCall):
        return calls
    if calls is None:
        calls = parse_tool_calls(content, tool_names)
    if calls is None:
        blocks = _JSON_FENCE.findall(content)
        if not blocks:
            return response
        calls = []
        for block in blocks:
            block_calls = parse_tool_calls(block, tool_names)
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
