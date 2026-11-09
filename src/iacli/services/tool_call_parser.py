"""Reconnaissance des appels d'outils écrits dans le texte d'un modèle (logique pure).

Certains petits modèles locaux n'utilisent pas le tool calling natif d'Ollama et
écrivent leur appel en JSON dans leur réponse."""

import json
import re
from collections.abc import Collection
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage

# Bloc de code markdown ```json ... ``` glissé par un petit modèle dans sa prose.
_JSON_FENCE = re.compile(r"```json[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


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
        calls.append(
            {
                "name": name,
                "args": arguments,
                "id": str(uuid4()),
                "type": "tool_call",
            }
        )
    return calls


def normalize_text_tool_call(response: Any, tool_names: Collection[str]) -> Any:
    """Convertit un appel d'outil écrit en JSON dans le texte (brut ou dans un bloc
    ```json) en vrai tool call ; sinon renvoie la réponse inchangée."""
    if not isinstance(response, AIMessage) or response.tool_calls:
        return response
    if not isinstance(response.content, str):
        return response
    content = response.content
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
