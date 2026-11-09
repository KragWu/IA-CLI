"""Objets de valeur décrivant la configuration de l'application."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# Fenêtre de contexte demandée à Ollama, en tokens.
DEFAULT_NUM_CTX = 18000


class EndOfLine(StrEnum):
    """Fin de ligne des nouveaux fichiers (clé `end_of_line` de la section `[files]`)."""

    AUTO = "auto"  # celle du système d'exploitation
    LF = "lf"
    CRLF = "crlf"


@dataclass(frozen=True, slots=True)
class FileSettings:
    """Réglages d'écriture de fichiers issus de `config.toml` (section `[files]`)."""

    end_of_line: EndOfLine = EndOfLine.AUTO


def resolve_newline(setting: EndOfLine, platform_newline: str) -> str:
    """Fin de ligne effective (`\\n` ou `\\r\\n`) pour un réglage.

    `platform_newline` est celle du système (`os.linesep`) ; une valeur autre que CRLF
    (LF, ou CR seul d'un ancien Mac) donne LF."""
    if setting is EndOfLine.CRLF:
        return "\r\n"
    if setting is EndOfLine.LF:
        return "\n"
    return "\r\n" if platform_newline == "\r\n" else "\n"


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    """Réglages Ollama issus de `config.toml` (section `[ollama]`)."""

    base_url: str  # instance Ollama, sans « / » final
    registry_url: str  # registre des modèles, sans « / » final
    default_model: str  # clé `model` : modèle utilisé par le chat
    num_ctx: int = DEFAULT_NUM_CTX  # fenêtre de contexte demandée, en tokens


@dataclass(frozen=True, slots=True)
class ConfigurationState:
    """Configuration chargée : dossier utilisé, réglages et avertissements.

    `warnings` n'est pas vide quand `config.toml` est illisible ; les réglages
    contiennent alors les valeurs par défaut."""

    directory: Path
    ollama: OllamaSettings
    warnings: tuple[str, ...] = ()
    files: FileSettings = field(default_factory=FileSettings)
