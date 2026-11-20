"""Fichier d'historique des messages, réservé à son propriétaire."""

import contextlib
import os
from datetime import datetime
from pathlib import Path

from iacli.domain.history import format_entry, parse_entries


class FileHistoryStore:
    """Historique dans un fichier au format de `prompt_toolkit`. Ne fait jamais échouer le chat :
    une erreur d'accès est ignorée (historique vide, entrée non enregistrée)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[str]:
        try:
            return parse_entries(
                self._path.read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            return []

    def append(self, text: str) -> None:
        with contextlib.suppress(OSError):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Le fichier contient en clair ce que l'utilisateur a saisi : mode 0600 (sans effet
            # sous Windows, où les droits du profil s'appliquent).
            descriptor = os.open(
                self._path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600
            )
            with os.fdopen(descriptor, "a", encoding="utf-8", newline="\n") as file:
                file.write(format_entry(text, datetime.now().isoformat(sep=" ")))
