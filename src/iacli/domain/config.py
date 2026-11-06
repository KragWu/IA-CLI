"""Configuration value objects used by the application."""

from dataclasses import dataclass
from pathlib import Path


# Fenêtre de contexte demandée à Ollama, en tokens.
DEFAULT_NUM_CTX = 18000


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    base_url: str
    registry_url: str
    default_model: str
    num_ctx: int = DEFAULT_NUM_CTX


@dataclass(frozen=True, slots=True)
class ConfigurationState:
    directory: Path
    ollama: OllamaSettings
    warnings: tuple[str, ...] = ()
