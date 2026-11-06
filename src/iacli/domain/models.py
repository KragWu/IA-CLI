"""Value objects describing an installed model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Informations annoncées par Ollama pour un modèle ; None signifie « inconnu »."""

    capabilities: frozenset[str] | None = None
