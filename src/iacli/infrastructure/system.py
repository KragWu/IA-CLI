"""Operating-system and disk gateways."""

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Callable


class LocalSystemGateway:
    def __init__(
        self,
        which: Callable[[str], str | None] | None = None,
    ) -> None:
        self._which = which or shutil.which

    def prerequisite_warnings(self) -> list[tuple[str, str]]:
        warnings: list[tuple[str, str]] = []
        if sys.version_info < (3, 11):
            warnings.append(("Python 3.11+", self._install_hint("python")))
        for executable, label in (("git", "Git"), ("rg", "ripgrep")):
            if self._which(executable) is None:
                warnings.append((label, self._install_hint(executable)))
        return warnings

    def available_model_disk_bytes(self) -> int:
        models_directory = Path(
            os.environ.get(
                "OLLAMA_MODELS",
                str(Path.home() / ".ollama" / "models"),
            )
        )
        path = models_directory
        while not path.exists() and path != path.parent:
            path = path.parent
        return shutil.disk_usage(path).free

    @staticmethod
    def _install_hint(executable: str) -> str:
        system_name = platform.system().lower()
        if "windows" in system_name:
            hints = {
                "python": "Installez vfox (gestionnaire de versions) : 'winget install version-fox.vfox'.\n" + 
                "Puis après redémarrage du terminal faites : 'vfox add python'.\n" +
                "Puis 'vfox install python@3.11' (ou supérieur) et pour finir 'vfox use -g python@3.11'.",
                "git": "Installer Git via 'winget install --id Git.Git -e' ou depuis git-scm.com.",
                "rg": "Installer ripgrep via 'winget install --id BurntSushi.ripgrep.MSBuild -e'.",
            }
        elif "darwin" in system_name or "mac" in system_name:
            hints = {
                "python": "Installer Python avec 'brew install python@3.11' (ou supérieur).\nPuis faites 'brew link --overwrite python@3.11' " +
                "pour définir la version par défaut.",
                "git": "Installer Git avec 'brew install git'.",
                "rg": "Installer ripgrep avec 'brew install ripgrep'.",
            }
        else:
            hints = {
                "python": "Installer SDKMAN (gestionnaire de version) 'curl -s \"https://get.sdkman.io\" | bash'.\n" +
                "Puis faites la commande 'sdk install python 3.11.0-open' (ou supérieur) et finir par 'sdk use python 3.11.0-open'",
                "git": "Installer Git avec 'sudo apt install git' (Debian/Ubuntu), 'sudo dnf install git' (Fedora) ou 'sudo apk add git' (Alpine).",
                "rg": "Installer ripgrep avec 'sudo apt install ripgrep' (Debian/Ubuntu), 'sudo dnf install ripgrep' (Fedora) ou 'sudo apk add ripgrep' (Alpine).",
            }
        return hints.get(executable, f"Installez '{executable}' pour votre système d'exploitation.")
