"""Point d'entrée de la commande iacli."""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

from iacli.adapters.cli_chat import CliChatPresenter
from iacli.adapters.cli_presenter import CliPresenter
from iacli.domain.config import OllamaSettings, resolve_newline
from iacli.infrastructure.configuration import FileConfigurationGateway
from iacli.infrastructure.ollama import OllamaClient
from iacli.infrastructure.system import LocalSystemGateway
from iacli.infrastructure.workspace_tools import create_workspace_tools
from iacli.services.chat import run_chat
from iacli.services.initialize import InitializeApplication
from iacli.services.ports import OllamaGateway


class OllamaClientFactory:
    """Fabrique d'`OllamaClient` à partir des réglages de `config.toml`."""

    def create(self, settings: OllamaSettings) -> OllamaGateway:
        return OllamaClient(
            base_url=settings.base_url,
            registry_url=settings.registry_url,
        )


def handle_init(args: argparse.Namespace) -> None:
    """Gestionnaire de la sous-commande init : câble le cas d'usage et l'exécute."""
    use_case = InitializeApplication(
        configuration=FileConfigurationGateway(),
        system=LocalSystemGateway(),
        ollama_factory=OllamaClientFactory(),
        presenter=CliPresenter(),
    )
    use_case.execute(args.config_dir)


def handle_chat(args: argparse.Namespace) -> None:
    """Lance une conversation avec le modèle configuré.

    Refuse de démarrer (exception affichée par `main`) si `config.toml` est illisible,
    si le modèle n'est pas installé ou s'il ne gère pas les outils. Un `num_ctx`
    supérieur au contexte maximal du modèle est plafonné avec un avertissement.
    """
    from langchain_ollama import ChatOllama

    configuration = FileConfigurationGateway().ensure_configuration(args.config_dir)
    if configuration.warnings:
        raise ValueError(
            "Configuration illisible : "
            + "; ".join(configuration.warnings)
            + ". Corrigez config.toml ou lancez iacli init."
        )
    ollama = OllamaClientFactory().create(configuration.ollama)
    installed_models = ollama.installed_models()
    model_name = configuration.ollama.default_model
    if model_name not in installed_models:
        raise RuntimeError(
            f"Le modèle {model_name} n'est pas installé. Lancez d'abord `iacli init`."
        )

    model_info = ollama.model_info(model_name)
    capabilities = model_info.capabilities
    if capabilities is not None and "tools" not in capabilities:
        raise RuntimeError(
            f"Le modèle {model_name} ne gère pas les outils, indispensables au chat. "
            "Choisissez un modèle compatible dans config.toml puis lancez `iacli init`."
        )

    requested_num_ctx = configuration.ollama.num_ctx
    num_ctx = model_info.limit_context(requested_num_ctx)
    if num_ctx != requested_num_ctx:
        CliPresenter().report_warning(
            f"num_ctx ({requested_num_ctx}) dépasse le contexte maximal "
            f"du modèle {model_name} ({num_ctx}) ; {num_ctx} sera utilisé."
        )

    instructions_path = configuration.directory / "IACLI.md"
    try:
        # utf-8-sig : le Bloc-notes ajoute un BOM aux fichiers UTF-8.
        instructions = instructions_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{instructions_path} n'est pas encodé en UTF-8. "
            "Enregistrez-le en UTF-8 puis relancez iacli."
        ) from error
    model = ChatOllama(
        model=model_name,
        base_url=configuration.ollama.base_url,
        temperature=0,
        num_ctx=num_ctx,
    )
    run_chat(
        model,
        Path.cwd().resolve(),
        instructions,
        model_name,
        presenter=CliChatPresenter(),
        # Fin de ligne des nouveaux fichiers : réglage `[files] end_of_line`, sinon l'OS.
        tools_factory=partial(
            create_workspace_tools,
            default_newline=resolve_newline(
                configuration.files.end_of_line, os.linesep
            ),
        ),
        context_limit=num_ctx,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser d'arguments de la CLI."""
    parser = argparse.ArgumentParser(
        prog="iacli", description="CLI locale pour les IA open source"
    )
    parser.set_defaults(func=handle_chat)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Chemin optionnel vers le dossier de configuration",
    )
    commands = parser.add_subparsers(dest="command")

    init_parser = commands.add_parser(
        "init", help="Initialiser la configuration globale"
    )
    init_parser.add_argument(
        "--config-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="Chemin optionnel vers le dossier de configuration",
    )
    init_parser.set_defaults(func=handle_init)

    chat_parser = commands.add_parser(
        "chat",
        help="Démarrer une conversation interactive",
    )
    chat_parser.add_argument(
        "--config-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="Chemin optionnel vers le dossier de configuration",
    )
    chat_parser.set_defaults(func=handle_chat)

    return parser


def main() -> int:
    """Exécute la CLI et retourne le code de sortie : 0 succès, 1 erreur, 130 interruption (Ctrl+C).

    Sans sous-commande, la conversation (`chat`) est lancée.
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        CliPresenter().report_interruption()
        return 130
    except Exception as err:
        CliPresenter().report_error(err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
