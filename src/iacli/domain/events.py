"""Application events presented by the outer adapters."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class InitializationEventCode(StrEnum):
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
    code: InitializationEventCode
    values: Mapping[str, str | int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    status: str
    completed: int | None = None
    total: int | None = None
