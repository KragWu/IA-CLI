"""Slash-commands du chat : registre et analyse d'une saisie, partagés par services et adapters."""

import re
from dataclasses import dataclass
from enum import StrEnum


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


class InputKind(StrEnum):
    """Nature d'une saisie du chat."""

    MESSAGE = "message"  # destinée au modèle
    COMMAND = "command"  # commande connue, seule
    COMMAND_WITH_ARGUMENTS = "command_with_arguments"  # commande connue suivie de texte
    UNKNOWN_COMMAND = "unknown_command"  # mot `/nom` inconnu, seul


@dataclass(frozen=True)
class ParsedInput:
    """Résultat de `parse_input` : la nature de la saisie et, pour une commande, son mot."""

    kind: InputKind
    word: str = ""  # premier mot tel que saisi (vide pour un message)
    command: SlashCommand | None = None  # commande enregistrée désignée par `word`


def parse_input(text: str) -> ParsedInput:
    """Classe une saisie ; c'est la seule règle de reconnaissance des commandes.

    Une commande connue (casse ignorée) doit être seule : avec du texte derrière, y compris
    sur d'autres lignes, ce n'est ni une commande ni un message, pour ne jamais perdre ce
    texte en silence. Un mot inconnu de la forme `/nom` n'est une erreur que s'il est seul ;
    `/tmp contient quoi ?` ou `/etc/hosts` sont des messages ordinaires."""
    words = text.split(maxsplit=1)
    if not words:
        return ParsedInput(InputKind.MESSAGE)
    word = words[0]
    has_arguments = len(words) > 1
    command = next((c for c in COMMANDS if c.name == word.casefold()), None)
    if command is not None:
        kind = InputKind.COMMAND_WITH_ARGUMENTS if has_arguments else InputKind.COMMAND
        return ParsedInput(kind, word, command)
    if not has_arguments and _COMMAND_WORD.fullmatch(word):
        return ParsedInput(InputKind.UNKNOWN_COMMAND, word)
    return ParsedInput(InputKind.MESSAGE)


def resolve_enter(text: str) -> tuple[str, bool]:
    """Effet d'Entrée sur `text` : (nouveau texte, True pour envoyer le message).

    Les barres obliques inverses en fin de texte vont par paires : chaque paire donne une barre
    littérale ; si une barre reste seule, elle demande de continuer sur la ligne suivante."""
    backslash = "\\"
    trailing = len(text) - len(text.rstrip(backslash))
    body = text[: len(text) - trailing]
    literal = backslash * (trailing // 2)
    if trailing % 2:
        return body + literal + "\n", False
    return body + literal, True
