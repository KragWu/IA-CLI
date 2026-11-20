"""Ports (contrats `Protocol`) dont les cas d'usage ont besoin.

Les services ne connaissent que ces contrats ; `cli.py` branche les implémentations
concrètes (infrastructure et adapters)."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from langchain_core.tools import BaseTool

from iacli.domain.commands import SlashCommand
from iacli.domain.config import ConfigurationState, OllamaSettings
from iacli.domain.events import DownloadProgress, InitializationEvent
from iacli.domain.models import ModelInfo


class ConfigurationGateway(Protocol):
    """Accès à la configuration globale."""

    def ensure_configuration(self, directory: Path | None = None) -> ConfigurationState:
        """Crée les fichiers manquants (sans écraser) puis charge la configuration.

        `directory` vaut `~/.config/iacli` par défaut."""
        ...


class SystemGateway(Protocol):
    """Informations sur la machine hôte."""

    def prerequisite_warnings(self) -> list[tuple[str, str]]:
        """Prérequis manquants, sous forme de paires (dépendance, conseil d'installation)."""
        ...

    def available_model_disk_bytes(self) -> int:
        """Octets libres à l'emplacement où Ollama stocke ses modèles."""
        ...


class OllamaGateway(Protocol):
    """Dialogue avec une instance Ollama et avec le registre des modèles.

    Les erreurs réseau remontent en `OSError`, les réponses invalides en `ValueError`."""

    def installed_models(self) -> set[str]:
        """Noms des modèles installés localement (ex. `qwen2.5-coder:7b`)."""
        ...

    def model_info(self, model: str) -> ModelInfo:
        """Informations annoncées par Ollama ; les champs inconnus valent None."""
        ...

    def remote_model_size(self, model: str) -> int:
        """Taille totale en octets du modèle dans le registre (`OSError` de code 404 s'il n'existe pas)."""
        ...

    def pull_model(
        self, model: str, progress: Callable[[DownloadProgress], None]
    ) -> None:
        """Télécharge le modèle en appelant `progress` à chaque mise à jour ; `ValueError` en cas d'échec."""
        ...


class OllamaGatewayFactory(Protocol):
    """Fabrique d'`OllamaGateway` : l'URL n'est connue qu'après lecture de la configuration."""

    def create(self, settings: OllamaSettings) -> OllamaGateway: ...


class InitializationPresenter(Protocol):
    """Sortie utilisateur de `iacli init`."""

    def present(self, event: InitializationEvent) -> None: ...

    def confirm_model_download(self, model: str, size_bytes: int) -> bool:
        """Demande à l'utilisateur s'il accepte le téléchargement."""
        ...


class ChatInput(Protocol):
    """Lecture bloquante de la saisie : propre à la boucle console (une interface pilotée par
    des événements, comme Textual, reçoit la saisie et appelle `ChatSession.handle`)."""

    def prompt(self) -> str:
        """Lit la saisie de l'utilisateur ; `EOFError` si l'entrée est fermée."""
        ...


class ChatPresenter(Protocol):
    """Affichage et confirmations d'une conversation, sans lecture de la saisie."""

    def start(self, workspace: Path, model: str) -> None: ...

    def show_response(self, response: str) -> None: ...

    def show_error(self, message: str) -> None: ...

    def show_notice(self, message: str) -> None:
        """Affiche une information neutre (retour d'une slash-command)."""
        ...

    def show_help(self, commands: Sequence[SlashCommand]) -> None: ...

    def show_cancelled(self) -> None:
        """Annonce que l'utilisateur a interrompu (Ctrl+C) la génération en cours."""
        ...

    def show_tool_call(self, name: str, arguments: dict[str, object]) -> None: ...

    def show_tool_result(self, name: str) -> None: ...

    def show_action_timing(self, action: str, seconds: float) -> None: ...

    def show_total_duration(self, seconds: float) -> None: ...

    def show_context_usage(self, used: int, limit: int) -> None: ...

    def show_context_warning(self, used: int, limit: int) -> None: ...

    def clear_context_usage(self) -> None:
        """Oublie l'occupation du contexte affichée (l'historique vient d'être effacé)."""
        ...

    def start_progress(self) -> None:
        """Démarre l'indicateur de traitement d'un tour."""
        ...

    def update_progress(self, message: str) -> None: ...

    def stop_progress(self) -> None: ...

    def confirm_file_write(self, path: str, diff: str) -> bool:
        """Affiche le diff et renvoie True si l'utilisateur accepte l'écriture."""
        ...

    def confirm_shell_command(self, command: str, workspace: Path) -> bool:
        """Affiche la commande et renvoie True si l'utilisateur accepte de l'exécuter."""
        ...


class HistoryStore(Protocol):
    """Historique persistant des messages saisis (ne lève jamais d'erreur d'accès)."""

    def load(self) -> list[str]: ...

    def append(self, text: str) -> None: ...


class ToolsFactory(Protocol):
    """Construit les outils de l'agent pour un projet.

    `confirm_*` sont appelés avant toute écriture ou commande ; `on_action_timing`
    reçoit (nom de l'outil, durée en secondes)."""

    def __call__(
        self,
        workspace: Path,
        confirm_file_write: Callable[[str, str], bool],
        confirm_shell_command: Callable[[str, Path], bool],
        on_action_timing: Callable[[str, float], None] | None = None,
    ) -> Sequence[BaseTool]: ...


class ConsolePresenter(ChatPresenter, ChatInput, Protocol):
    """Presenter de la boucle console : affiche, demande confirmation et lit la saisie."""
