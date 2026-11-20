"""Historique des messages saisis : navigation et format du fichier (logique pure)."""

from collections.abc import Iterable


class InputHistory:
    """Messages déjà saisis, parcourus avec les flèches ; le brouillon en cours est restitué
    en revenant à la fin."""

    def __init__(self, entries: Iterable[str] = ()) -> None:
        self._entries = list(entries)
        self._position = len(self._entries)
        self._draft = ""

    def add(self, text: str) -> bool:
        """Mémorise un message (jamais vide ni identique au précédent) et repart de la fin."""
        self._position = len(self._entries) + 1
        stored = bool(text.strip()) and (not self._entries or self._entries[-1] != text)
        if stored:
            self._entries.append(text)
        self._position = len(self._entries)
        return stored

    def previous(self, current: str) -> str | None:
        """Message plus ancien, ou `None` au début ; `current` est gardé comme brouillon."""
        if self._position == 0:
            return None
        if self._position == len(self._entries):
            self._draft = current
        self._position -= 1
        return self._entries[self._position]

    def next(self) -> str | None:
        """Message plus récent (puis le brouillon), ou `None` quand on est déjà à la fin."""
        if self._position >= len(self._entries):
            return None
        self._position += 1
        if self._position == len(self._entries):
            return self._draft
        return self._entries[self._position]


def format_entry(text: str, timestamp: str) -> str:
    """Une entrée du fichier d'historique, au format de `prompt_toolkit` (partagé avec le chat
    console) : une ligne `# horodatage` puis chaque ligne du message préfixée par `+`."""
    lines = "".join(f"+{line}\n" for line in text.split("\n"))
    return f"\n# {timestamp}\n{lines}"


def parse_entries(content: str) -> list[str]:
    """Messages d'un fichier d'historique, du plus ancien au plus récent."""
    entries: list[list[str]] = []
    current: list[str] | None = None
    for line in content.splitlines():
        if line.startswith("+"):
            if current is None:
                current = []
                entries.append(current)
            current.append(line[1:])
        else:
            current = None
    return ["\n".join(lines) for lines in entries]
