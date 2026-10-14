"""Point d'entrée de la commande iacli."""

import argparse
from pathlib import Path
import sys

from iacli.adapters.cli_presenter import CliPresenter
from iacli.domain.config import OllamaSettings
from iacli.infrastructure.configuration import FileConfigurationGateway
from iacli.infrastructure.ollama import OllamaClient
from iacli.infrastructure.system import LocalSystemGateway
from iacli.services.chat import run_chat
from iacli.services.initialize import InitializeApplication
from iacli.services.ports import OllamaGateway


class OllamaClientFactory:
    def create(self, settings: OllamaSettings) -> OllamaGateway:
        return OllamaClient(
            base_url=settings.base_url,
            registry_url=settings.registry_url,
        )


def handle_init(args: argparse.Namespace) -> None:
    """Gestionnaire de la sous-commande init."""
    use_case = InitializeApplication(
        configuration=FileConfigurationGateway(),
        system=LocalSystemGateway(),
        ollama_factory=OllamaClientFactory(),
        presenter=CliPresenter(),
    )
    use_case.execute(args.config_dir)


def handle_chat(args: argparse.Namespace) -> None:
    """Lance une conversation avec le modèle configuré."""
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

    instructions_path = configuration.directory / "IACLI.md"
    instructions = instructions_path.read_text(encoding="utf-8")
    model = ChatOllama(
        model=model_name,
        base_url=configuration.ollama.base_url,
        temperature=0,
    )
    run_chat(model, Path.cwd().resolve(), instructions, model_name)


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser d'arguments de la CLI."""
    parser = argparse.ArgumentParser(
        prog="iacli",
        description="CLI locale pour les IA open source"
    )
    parser.set_defaults(func=handle_chat, config_dir=None)
    commands = parser.add_subparsers(dest="command")

    init_parser = commands.add_parser(
        "init",
        help="Initialiser la configuration globale"
    )
    init_parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Chemin optionnel vers le dossier de configuration"
    )
    init_parser.set_defaults(func=handle_init)

    chat_parser = commands.add_parser(
        "chat",
        help="Démarrer une conversation interactive",
    )
    chat_parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Chemin optionnel vers le dossier de configuration",
    )
    chat_parser.set_defaults(func=handle_chat)

    return parser


def main() -> int:
    """Exécute la CLI et retourne le code de statut système."""
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
