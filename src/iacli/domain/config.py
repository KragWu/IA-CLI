"""Configuration value objects used by the application."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    base_url: str
    registry_url: str
    default_model: str


@dataclass(frozen=True, slots=True)
class ConfigurationState:
    directory: Path
    ollama: OllamaSettings
    warnings: tuple[str, ...] = ()
