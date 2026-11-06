"""Value objects describing an installed model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Informations annoncées par Ollama pour un modèle ; None signifie « inconnu »."""

    capabilities: frozenset[str] | None = None
    context_length: int | None = None

    def limit_context(self, requested: int) -> int:
        """Fenêtre de contexte utilisable : celle demandée, plafonnée au maximum du modèle."""
        if self.context_length is None:
            return requested
        return min(requested, self.context_length)
