"""Slash-commands du chat : registre partagé par les services et l'autocomplétion."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str


COMMANDS = (
    SlashCommand("/help", "Affiche la liste des commandes"),
    SlashCommand("/clear", "Efface l'historique de la conversation"),
    SlashCommand("/context", "Affiche l'occupation du contexte du dernier tour"),
    SlashCommand("/exit", "Termine la conversation"),
    SlashCommand("/quit", "Termine la conversation (comme /exit)"),
)

# Un mot qui ressemble à une commande : `/etc/hosts` n'en est pas une (chemin).
_COMMAND_WORD = re.compile(r"/[a-z][a-z0-9_-]*", re.IGNORECASE)


def _first_word(text: str) -> str:
    words = text.split(maxsplit=1)
    return words[0] if words else ""


def looks_like_command(text: str) -> bool:
    """True si la saisie commence par un mot de la forme `/nom`."""
    return _COMMAND_WORD.fullmatch(_first_word(text)) is not None


def is_command_input(text: str) -> bool:
    """True si la saisie est destinée aux commandes plutôt qu'au modèle.

    Une commande connue, avec ou sans texte derrière, en est une. Un mot inconnu de la forme
    `/nom` n'en est une que seul : `/tmp contient quoi ?` est un message ordinaire."""
    if find_command(text) is not None:
        return True
    return looks_like_command(text) and len(text.split()) == 1


def find_command(text: str) -> SlashCommand | None:
    """Commande enregistrée désignée par le premier mot de `text` (casse ignorée)."""
    word = _first_word(text).casefold()
    return next((command for command in COMMANDS if command.name == word), None)
