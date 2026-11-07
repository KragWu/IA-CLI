"""Shim de compatibilité : l'ancien chemin d'import du client Ollama.

Le code vit désormais dans `iacli.infrastructure.ollama` ; ce module le ré-exporte."""

from iacli.infrastructure.ollama import OllamaClient

__all__ = ["OllamaClient"]
