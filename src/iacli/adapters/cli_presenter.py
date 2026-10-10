"""Console presenter and prompt adapter."""

import sys
from typing import Callable

from iacli.domain.events import InitializationEvent, InitializationEventCode


class CliPresenter:
    _steps = {
        1: "vérification des prérequis système",
        2: "création de la configuration globale",
        3: "vérification de la connexion à Ollama",
        4: "vérification du modèle et de l'espace disque",
    }

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        error_output_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._input = input_fn
        self._output = output_fn
        self._error_output = error_output_fn or (
            lambda message: print(message, file=sys.stderr)
        )

    def present(self, event: InitializationEvent) -> None:
        code = event.code
        values = event.values
        if code is InitializationEventCode.STEP:
            number = values["number"]
            label = self._steps[number]
            suffix = f" ({values['url']})" if "url" in values else ""
            self._output(f"Étape {number}/4 : {label}{suffix}.")
        elif code is InitializationEventCode.PREREQUISITE_MISSING:
            self._output(f"Avertissement : dépendance système manquante : {values['dependency']}.")
            self._output(f"Conseil : {values['hint']}")
        elif code is InitializationEventCode.CONFIGURATION_WARNING:
            self._output(f"Avertissement : configuration illisible ({values['error']}), utilisation des valeurs par défaut.")
        elif code is InitializationEventCode.OLLAMA_UNREACHABLE:
            self._output(f"Avertissement : Ollama n'est pas accessible sur {values['url']} ({values['error']}).")
        elif code is InitializationEventCode.OLLAMA_CONNECTED:
            self._output("Connexion à Ollama confirmée.")
        elif code is InitializationEventCode.MODEL_INSTALLED:
            self._output(f"Le modèle {values['model']} est déjà installé, aucun téléchargement nécessaire.")
        elif code is InitializationEventCode.MODEL_MISSING:
            self._output(f"Le modèle {values['model']} n'est pas présent localement.")
        elif code is InitializationEventCode.MODEL_LOOKUP_FAILED:
            self._output(f"Avertissement : impossible de vérifier le modèle dans le registre Ollama ({values['error']}).")
        elif code is InitializationEventCode.MODEL_NOT_FOUND:
            self._output(f"Avertissement : le modèle {values['model']} n'existe pas dans le registre Ollama.")
        elif code is InitializationEventCode.MODEL_FOUND:
            self._output(
                f"Modèle {values['model']} trouvé dans le registre Ollama "
                f"({self._format_size(values['size_bytes'])})."
            )
        elif code is InitializationEventCode.DISK_SPACE_INSUFFICIENT:
            self._output(
                "Avertissement : espace disque insuffisant pour le modèle "
                f"{values['model']} ({self._format_size(values['required_bytes'])} requis, "
                f"{self._format_size(values['available_bytes'])} disponibles)."
            )
        elif code is InitializationEventCode.PULL_STARTED:
            self._output(f"Téléchargement de {values['model']} lancé, progression reçue d'Ollama :")
        elif code is InitializationEventCode.PULL_PROGRESS:
            completed = values.get("completed")
            total = values.get("total")
            if isinstance(completed, int) and isinstance(total, int) and total > 0:
                percentage = min(100, completed * 100 // total)
                self._output(
                    f"{values['status']} : {percentage}% "
                    f"({completed / 1024**2:.1f}/{total / 1024**2:.1f} Mio)"
                )
            else:
                self._output(f"Téléchargement : {values['status']}")
        elif code is InitializationEventCode.PULL_FAILED:
            self._output(f"Avertissement : échec du téléchargement ({values['error']}).")
        elif code is InitializationEventCode.PULL_COMPLETED:
            self._output(f"Modèle {values['model']} téléchargé.")
        elif code is InitializationEventCode.PULL_SKIPPED:
            self._output(f"Téléchargement du modèle {values['model']} ignoré.")
        elif code is InitializationEventCode.CONFIGURATION_CREATED:
            self._output(f"Configuration créée dans {values['directory']}.")
        elif code is InitializationEventCode.INITIALIZATION_COMPLETED:
            self._output("Initialisation terminée.")

    def report_interruption(self) -> None:
        self._error_output("[iacli] Opération annulée par l'utilisateur.")

    def report_error(self, error: Exception) -> None:
        self._error_output(f"[iacli] Erreur : {error}")

    def confirm_model_download(self, model: str, size_bytes: int) -> bool:
        answer = self._input(
            f"Télécharger le modèle {model} ({self._format_size(size_bytes)}) ? [O/n] "
        ).strip().lower()
        return answer in ("", "o", "oui", "y", "yes")

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        gib = size_bytes / 1024**3
        if gib >= 1:
            return f"{gib:.2f} Gio"
        return f"{size_bytes / 1024**2:.0f} Mio"
