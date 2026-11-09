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


def _make_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": arguments, "id": str(uuid4()), "type": "tool_call"}


_TAG_EXPECTED_FORMAT = (
    'Renvoie un JSON valide de la forme {"name": "<outil>", "arguments": {...}} '
    "entre <tool_call> et </tool_call>."
)


@dataclass(frozen=True)
class MalformedToolCall:
    """Appel d'outil manifestement raté (balise `<tool_call>` illisible).

    `error` est un message destiné au modèle pour qu'il corrige son appel."""

    error: str


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


def _malformed(reason: str) -> MalformedToolCall:
    return MalformedToolCall(
        f"Ton appel d'outil <tool_call> est invalide : {reason} {_TAG_EXPECTED_FORMAT}"
    )


def _skip_spaces(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _parse_tagged_tool_calls(
    text: str,
) -> list[dict[str, Any]] | MalformedToolCall | None:
    """Renvoie les tool calls des balises `<tool_call>` de `text`, un `MalformedToolCall`
    si l'une est invalide, None s'il n'y en a aucune. Le nom n'est pas confronté aux
    outils connus : `ToolNode` renverra lui-même une erreur au modèle pour un outil
    inconnu.

    Une balise n'est une tentative d'appel que si du JSON (`{` ou `[`) la suit ; sinon
    c'est une simple mention dans la prose, ignorée. Le JSON est lu par un décodeur et
    non par une regex : ses chaînes peuvent donc contenir `</tool_call>`."""
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    position = 0
    while (opening := text.find(_TOOL_CALL_OPEN, position)) != -1:
        start = _skip_spaces(text, opening + len(_TOOL_CALL_OPEN))
        if text[start : start + 1] not in ("{", "["):
            position = start
            continue
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError as error:
            return _malformed(f"JSON illisible ({error}).")
        end = _skip_spaces(text, end)
        if not text.startswith(_TOOL_CALL_CLOSE, end):
            return _malformed(
                "une balise <tool_call> n'est pas refermée par </tool_call>."
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
    calls = _parse_tagged_tool_calls(content)
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
