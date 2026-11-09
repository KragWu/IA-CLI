"""Configuration globale stockée dans des fichiers (`config.toml` et `IACLI.md`)."""

import tomllib
from pathlib import Path

from iacli.domain.config import (
    DEFAULT_NUM_CTX,
    ConfigurationState,
    EndOfLine,
    FileSettings,
    OllamaSettings,
)

# Contenu écrit dans `config.toml` à la première exécution.
DEFAULT_CONFIG = """[ollama]
base_url = "http://localhost:11434"
registry_url = "https://registry.ollama.ai"
model = "qwen2.5-coder:7b"
# Fenêtre de contexte demandée à Ollama (en tokens). Elle ne doit pas dépasser
# le contexte maximal du modèle choisi.
num_ctx = 18000

[files]
# Fin de ligne des nouveaux fichiers : "auto" (celle du système), "lf" ou "crlf".
# Un fichier existant garde sa fin de ligne la plus fréquente.
end_of_line = "auto"
"""

# Contenu initial de `IACLI.md`, injecté dans l'invite système du chat.
DEFAULT_INSTRUCTIONS = """# Instructions globales IA-CLI

- Répondre en français par défaut, sauf demande contraire.
- Privilégier des réponses précises, utiles et vérifiables.
- Respecter les conventions du projet en cours.
"""

# Valeurs par défaut des clés textuelles de [ollama] ; `num_ctx` est géré à part.
_DEFAULT_OLLAMA = {
    "base_url": "http://localhost:11434",
    "registry_url": "https://registry.ollama.ai",
    "model": "qwen2.5-coder:7b",
}
# Valeurs renvoyées quand `config.toml` est illisible (`num_ctx` ajouté à part).
_DEFAULTS: dict[str, str | int] = {
    **_DEFAULT_OLLAMA,
    "end_of_line": EndOfLine.AUTO.value,
}


class FileConfigurationGateway:
    """Lit et initialise la configuration dans `~/.config/iacli` (ou le dossier fourni)."""

    def ensure_configuration(self, directory: Path | None = None) -> ConfigurationState:
        """Crée dossier et fichiers manquants sans jamais écraser l'existant, puis charge `config.toml`.

        Un fichier invalide ne lève pas d'exception : les valeurs par défaut sont
        utilisées et la cause est indiquée dans `ConfigurationState.warnings`."""
        target = directory or Path.home() / ".config" / "iacli"
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / "config.toml"
        instructions_path = target / "IACLI.md"
        if not config_path.exists():
            config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
        if not instructions_path.exists():
            instructions_path.write_text(DEFAULT_INSTRUCTIONS, encoding="utf-8")

        values, warnings = self._load_values(config_path)
        settings = OllamaSettings(
            base_url=str(values["base_url"]).rstrip("/"),
            registry_url=str(values["registry_url"]).rstrip("/"),
            default_model=str(values["model"]),
            num_ctx=int(values["num_ctx"]),
        )
        files = FileSettings(end_of_line=EndOfLine(str(values["end_of_line"])))
        return ConfigurationState(target, settings, tuple(warnings), files)

    @staticmethod
    def _load_values(config_path: Path) -> tuple[dict[str, str | int], list[str]]:
        """Valide les sections [ollama] et [files] ; renvoie (valeurs, avertissements).

        À la première erreur, toutes les valeurs par défaut sont renvoyées avec le message d'erreur."""
        try:
            with config_path.open("rb") as config_file:
                document = tomllib.load(config_file)
            if not isinstance(document, dict):
                raise ValueError("Le contenu TOML doit être une table.")
            ollama = document.get("ollama", {})
            if not isinstance(ollama, dict):
                raise ValueError("La section [ollama] doit être une table TOML.")
            values: dict[str, str | int] = {}
            for key, default in _DEFAULT_OLLAMA.items():
                value = ollama.get(key, default)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"La valeur ollama.{key} doit être une chaîne non vide."
                    )
                values[key] = value.strip()
            num_ctx = ollama.get("num_ctx", DEFAULT_NUM_CTX)
            if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx < 1:
                raise ValueError(
                    "La valeur ollama.num_ctx doit être un entier strictement positif."
                )
            values["num_ctx"] = num_ctx
            files = document.get("files", {})
            if not isinstance(files, dict):
                raise ValueError("La section [files] doit être une table TOML.")
            end_of_line = files.get("end_of_line", EndOfLine.AUTO.value)
            choices = [choice.value for choice in EndOfLine]
            if (
                not isinstance(end_of_line, str)
                or end_of_line.strip().lower() not in choices
            ):
                quoted = ", ".join(f'"{choice}"' for choice in choices)
                raise ValueError(
                    f"La valeur files.end_of_line doit être l'une de : {quoted}."
                )
            values["end_of_line"] = end_of_line.strip().lower()
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            return {**_DEFAULTS, "num_ctx": DEFAULT_NUM_CTX}, [str(error)]
        return values, []
