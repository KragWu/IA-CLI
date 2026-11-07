"""Événements métier émis par les cas d'usage et mis en forme par les adapters."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class InitializationEventCode(StrEnum):
    """Type d'événement émis pendant `iacli init`."""

    STEP = "step"
    PREREQUISITE_MISSING = "prerequisite_missing"
    CONFIGURATION_WARNING = "configuration_warning"
    OLLAMA_UNREACHABLE = "ollama_unreachable"
    OLLAMA_CONNECTED = "ollama_connected"
    MODEL_INSTALLED = "model_installed"
    MODEL_MISSING = "model_missing"
    MODEL_LOOKUP_FAILED = "model_lookup_failed"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_FOUND = "model_found"
    DISK_SPACE_INSUFFICIENT = "disk_space_insufficient"
    PULL_STARTED = "pull_started"
    PULL_PROGRESS = "pull_progress"
    PULL_FAILED = "pull_failed"
    PULL_COMPLETED = "pull_completed"
    PULL_SKIPPED = "pull_skipped"
    CONFIGURATION_CREATED = "configuration_created"
    INITIALIZATION_COMPLETED = "initialization_completed"


@dataclass(frozen=True, slots=True)
class InitializationEvent:
    """Événement d'initialisation : un code et ses valeurs (modèle, url, erreur…).

    Le texte affiché à l'utilisateur est décidé par le présentateur, pas ici."""

    code: InitializationEventCode
    values: Mapping[str, str | int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    """Progression d'un téléchargement de modèle ; `completed`/`total` en octets, None si inconnus."""

    status: str
    completed: int | None = None
    total: int | None = None
