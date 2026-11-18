"""État affiché par la saisie du chat (barre d'état), partagé entre les adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptStatus:
    """Ce que la barre d'état de la saisie affiche : modèle et occupation du contexte."""

    model: str
    context_used: int | None
    context_limit: int | None


def format_tokens(count: int) -> str:
    """Nombre de tokens lisible, milliers séparés par une espace (`18 000`)."""
    return f"{count:,}".replace(",", " ")
