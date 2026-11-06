"""Filesystem-backed configuration gateway."""

import tomllib
from pathlib import Path

from iacli.domain.config import DEFAULT_NUM_CTX, ConfigurationState, OllamaSettings

DEFAULT_CONFIG = """[ollama]
base_url = "http://localhost:11434"
registry_url = "https://registry.ollama.ai"
model = "qwen2.5-coder:7b"
# Fenêtre de contexte demandée à Ollama (en tokens). Elle ne doit pas dépasser
# le contexte maximal du modèle choisi.
num_ctx = 18000
"""

DEFAULT_INSTRUCTIONS = """# Instructions globales IA-CLI

- Répondre en français par défaut, sauf demande contraire.
- Privilégier des réponses précises, utiles et vérifiables.
- Respecter les conventions du projet en cours.
"""

_DEFAULT_OLLAMA = {
    "base_url": "http://localhost:11434",
    "registry_url": "https://registry.ollama.ai",
    "model": "qwen2.5-coder:7b",
}


class FileConfigurationGateway:
    def ensure_configuration(self, directory: Path | None = None) -> ConfigurationState:
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
            base_url=values["base_url"].rstrip("/"),
            registry_url=values["registry_url"].rstrip("/"),
            default_model=values["model"],
            num_ctx=values["num_ctx"],
        )
        return ConfigurationState(target, settings, tuple(warnings))

    @staticmethod
    def _load_values(config_path: Path) -> tuple[dict[str, str | int], list[str]]:
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
                    raise ValueError(f"La valeur ollama.{key} doit être une chaîne non vide.")
                values[key] = value.strip()
            num_ctx = ollama.get("num_ctx", DEFAULT_NUM_CTX)
            if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx < 1:
                raise ValueError(
                    "La valeur ollama.num_ctx doit être un entier strictement positif."
                )
            values["num_ctx"] = num_ctx
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            return {**_DEFAULT_OLLAMA, "num_ctx": DEFAULT_NUM_CTX}, [str(error)]
        return values, []
