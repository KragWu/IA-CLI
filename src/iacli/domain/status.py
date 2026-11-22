"""État affiché par la saisie du chat (barre d'état), partagé entre les adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptStatus:
    """Ce que la barre d'état de la saisie affiche : modèle et occupation du contexte."""

    model: str
    context_used: int | None
    context_limit: int | None


# Part de la fenêtre de contexte à partir de laquelle l'occupation passe à l'orange / au rouge.
WARN_RATIO = 0.6
DANGER_RATIO = 0.8


def context_percent(used: int, limit: int) -> int:
    """Occupation du contexte en pourcentage entier (0 si la limite est nulle)."""
    return round(100 * used / limit) if limit > 0 else 0


def context_level(used: int, limit: int) -> str:
    """Niveau d'occupation du contexte : `ok`, `warn` (dès 60 %) ou `danger` (dès 80 %)."""
    ratio = used / limit if limit > 0 else 0
    if ratio >= DANGER_RATIO:
        return "danger"
    return "warn" if ratio >= WARN_RATIO else "ok"


def format_tokens(count: int) -> str:
    """Nombre de tokens lisible, milliers séparés par une espace (`18 000`)."""
    return f"{count:,}".replace(",", " ")


def format_duration(seconds: float) -> str:
    """Durée lisible : `850 ms`, `1.25 s`, `2 min 05 s` ou `1 h 02 min 03 s`."""
    # Les seuils suivent la valeur arrondie : 59,996 s s'affiche `1 min 00 s`, pas `60.00 s`.
    if seconds < 0.9995:
        milliseconds = seconds * 1000
        if milliseconds < 10:
            return f"{milliseconds:.1f} ms"
        return f"{milliseconds:.0f} ms"
    if seconds < 59.995:
        return f"{seconds:.2f} s"
    whole_seconds = round(seconds)
    if whole_seconds < 3600:
        minutes, remaining_seconds = divmod(whole_seconds, 60)
        return f"{minutes} min {remaining_seconds:02d} s"
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours} h {minutes:02d} min {remaining_seconds:02d} s"
