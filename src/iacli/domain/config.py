"""Objets de valeur décrivant la configuration de l'application."""

from dataclasses import dataclass
from pathlib import Path


# Fenêtre de contexte demandée à Ollama, en tokens.
DEFAULT_NUM_CTX = 18000


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
