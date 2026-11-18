"""État affiché par la saisie du chat (barre d'état), partagé entre les adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptStatus:
    """Ce que la barre d'état de la saisie affiche : modèle et occupation du contexte."""

    model: str
    context_used: int | None
    context_limit: int | None
